curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "any",
        "messages": [{"role":"user","content":"안녕! 한 줄 인사만 해줘"}],
        "stream": false
      }'
