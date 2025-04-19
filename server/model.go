package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	"text/template/parse"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var intermediateBlobs map[string]string = make(map[string]string)

type layerGGML struct {
	Layer
	*ggml.GGML
}

func parseFromModel(ctx context.Context, name model.Name, fn func(api.ProgressResponse)) (layers []*layerGGML, err error) {
	m, err := ParseNamedManifest(name)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := PullModel(ctx, name.String(), &registryOptions{}, fn); err != nil {
			return nil, err
		}

		m, err = ParseNamedManifest(name)
		if err != nil {
			return nil, err
		}
	case err != nil:
		return nil, err
	}

	for _, layer := range m.Layers {
		layer, err := NewLayerFromLayer(layer.Digest, layer.MediaType, name.DisplayShortest())
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter":
			blobpath, err := GetBlobsPath(layer.Digest)
			if err != nil {
				return nil, err
			}

			blob, err := os.Open(blobpath)
			if err != nil {
				return nil, err
			}
			defer blob.Close()

			f, _, err := ggml.Decode(blob, 0)
			if err != nil {
				return nil, err
			}

			layers = append(layers, &layerGGML{layer, f})
		default:
			layers = append(layers, &layerGGML{layer, nil})
		}
	}

	return layers, nil
}

func detectChatTemplate(layers []*layerGGML) ([]*layerGGML, error) {
	for _, layer := range layers {
		if s := layer.GGML.KV().ChatTemplate(); s != "" {
			if t, err := template.Named(s); err != nil {
				slog.Debug("template detection", "error", err, "template", s)
			} else {
				layer, err := NewLayer(t.Reader(), "application/vnd.ollama.image.template")
				if err != nil {
					return nil, err
				}

				layer.status = fmt.Sprintf("using autodetected template %s", t.Name)
				layers = append(layers, &layerGGML{layer, nil})

				if t.Parameters != nil {
					var b bytes.Buffer
					if err := json.NewEncoder(&b).Encode(t.Parameters); err != nil {
						return nil, err
					}

					layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
					if err != nil {
						return nil, err
					}

					layers = append(layers, &layerGGML{layer, nil})
				}
			}
		}
	}

	return layers, nil
}

func detectContentType(r io.Reader) (string, error) {
	var b bytes.Buffer
	if _, err := io.Copy(&b, r); err != nil {
		return "", err
	}

	if contentType := ggml.DetectContentType(b.Bytes()); contentType != "" {
		return contentType, nil
	}

	if contentType := http.DetectContentType(b.Bytes()); contentType != "application/octet-stream" {
		return contentType, nil
	}

	return "unknown", nil
}

func parseObjects(s string) []map[string]any {
	var objs []map[string]any
	for offset := 0; offset < len(s); {
		var obj map[string]any
		decoder := json.NewDecoder(strings.NewReader(s[offset:]))
		if err := decoder.Decode(&obj); errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		} else if syntax := &(json.SyntaxError{}); errors.As(err, &syntax) {
			// skip over any syntax errors
			offset += int(syntax.Offset)
		} else if unmarshalType := &(json.UnmarshalTypeError{}); errors.As(err, &unmarshalType) {
			// skip over any unmarshalable types
			offset += int(unmarshalType.Offset)
		} else if err != nil {
			return nil
		} else {
			offset += int(decoder.InputOffset())
			objs = append(objs, obj)
		}
	}

	return objs
}

// parseToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// mxyng: this only really works if the input contains tool calls in some JSON format
func (m *Model) parseToolCalls(s string) ([]api.ToolCall, bool) {
    // Sanitize JSON quotes first
    sanitized := replaceJSONQuotes(s)

    // Original template detection logic
    tmpl := m.Template.Subtree(func(n parse.Node) bool {
        if t, ok := n.(*parse.RangeNode); ok {
            return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
        }
        return false
    })

    if tmpl == nil {
        return nil, false
    }

    // Execute template to find key names
    var b bytes.Buffer
    if err := tmpl.Execute(&b, map[string][]api.ToolCall{
        "ToolCalls": {{
            Function: api.ToolCallFunction{
                Name: "@@name@@",
                Arguments: api.ToolCallFunctionArguments{
                    "@@argument@@": 1,
                },
            },
        }},
    }); err != nil {
        return nil, false
    }

    templateObjects := parseObjects(b.String())
    if len(templateObjects) == 0 {
        return nil, false
    }

    // Detect name/arguments keys
    var nameKey, argsKey string
    for k, v := range templateObjects[0] {
        switch v.(type) {
        case string:
            nameKey = k
        case map[string]any:
            argsKey = k
        }
    }

    if nameKey == "" || argsKey == "" {
        return nil, false
    }

    // Parse the sanitized response
    responseObjects := parseObjects(sanitized)
    if len(responseObjects) == 0 {
        return nil, false
    }

    // Collect nested objects
    var collect func(any) []map[string]any
    collect = func(obj any) (all []map[string]any) {
        switch o := obj.(type) {
        case map[string]any:
            all = append(all, o)
            for _, v := range o {
                all = append(all, collect(v)...)
            }
        case []any:
            for _, v := range o {
                all = append(all, collect(v)...)
            }
        }
        return
    }

    var objs []map[string]any
    for _, p := range responseObjects {
        objs = append(objs, collect(p)...)
    }

    var toolCalls []api.ToolCall
    for _, kv := range objs {
        // Try standard fields first
        name, _ := kv[nameKey].(string)
        args, _ := kv[argsKey].(map[string]any)

        // Fallback to model-specific fields
        if name == "" {
            name, _ = kv["tool_name"].(string)
        }
        if len(args) == 0 {
            args, _ = kv["tool_arguments"].(map[string]any)
        }

        if name != "" && len(args) > 0 {
            toolCalls = append(toolCalls, api.ToolCall{
                Function: api.ToolCallFunction{
                    Name:      name,
                    Arguments: args,
                },
            })
        }
    }

    return toolCalls, len(toolCalls) > 0
}

// Helper function for safe quote replacement
func replaceJSONQuotes(s string) string {
    var buf strings.Builder
    stack := 0
    inString := false
    escape := false

    for _, r := range s {
        switch {
        case !escape && r == '\\':
            escape = true
        case !escape && r == '"':
            inString = !inString
        case !escape && r == '{' && !inString:
            stack++
        case !escape && r == '}' && !inString:
            stack--
        }

        // Replace single quotes only in JSON structures
        if stack > 0 && !inString && r == '\'' {
            buf.WriteRune('"')
        } else {
            buf.WriteRune(r)
        }

        if escape {
            escape = false
        }
    }
    return buf.String()
}
