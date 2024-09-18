# llm_structure_pure
基于lmdeploy的纯净后端版
```
docker compose -f docker-compose-jinghu.yml build --no-cache

docker compose -f docker-compose-jinghu.yml up

# 如果想取消
docker-compose down

# 删除无标签的内容
docker images -f "dangling=true" -q | xargs docker rmi
```