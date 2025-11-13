
## API端点测试

### 健康检查
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET
```

### 创建用户
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/users" -Method POST -Body '{"username":"testuser","email":"test@example.com","password":"123456"}' -ContentType "application/json"
```

### 获取用户列表
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/users" -Method GET
```

### 获取特定用户
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/users/{user_id}" -Method GET
```

### 更新用户
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/users/{user_id}" -Method PUT -Body '{"username":"updateduser"}' -ContentType "application/json"
```

### 删除用户
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/users/{user_id}" -Method DELETE
```

### 获取系统统计
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/stats" -Method GET
```

## 启动服务器
```bash
python restful_api设计原则演示.py
```

## 访问文档
- API文档: http://localhost:8000/docs
- 备用文档: http://localhost:8000/redoc
