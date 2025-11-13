from fastapi import FastAPI, HTTPException, status, Depends, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging
import uuid
from uuid import UUID

# RESTful API设计原则9: 持续监控与日志记录
# 详细日志记录便于问题追踪和分析，确保API稳定性和可靠性
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RESTful API设计原则1: 资源定位与URI设计
# 使用直观简洁的层级结构URI标识资源 (/users/{id}访问特定用户)
# RESTful API设计原则5: 版本管理  
# API版本控制确保向后兼容性 (/api/v1/前缀)
app = FastAPI(
    title="RESTful API设计原则演示",
    description="演示RESTful API的10大设计原则",
    version="1.0.0",
    docs_url="/docs"
)

# RESTful API设计原则6: 安全性考虑
# CORS配置防止跨域攻击，HTTPS应在生产环境启用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 安全认证
security = HTTPBearer()

# 枚举定义
class UserRole(str, Enum):
    admin = "admin"
    user = "user"

# 基础响应模型
class APIResponse(BaseModel):
    success: bool = True
    message: str = "操作成功"
    data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(APIResponse):
    success: bool = False
    error_code: str

# 用户模型
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, description="用户名")
    email: str = Field(..., description="邮箱地址")
    password: str = Field(..., min_length=6, description="密码")

class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    role: UserRole = UserRole.user
    created_at: datetime

# 模拟数据库
users_db = {}

# RESTful API设计原则3: 状态管理与无状态性
# 每次请求包含所有必要信息，不依赖会话状态，保证服务独立性和可伸缩性
def get_current_user():
    """获取当前用户（模拟认证）"""
    # 实际应用中这里应该验证JWT token
    return {"id": UUID("12345678-1234-1234-1234-123456789012"), "role": "user"}

def require_role(role: UserRole):
    """角色权限检查"""
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") != role.value:
            raise HTTPException(status_code=403, detail="权限不足")
        return current_user
    return role_checker

# RESTful API设计原则7: 文档化与工具支持
# FastAPI自动生成OpenAPI文档，提供详尽的API说明和交互式测试界面
@app.get("/", tags=["系统"])
async def root():
    """健康检查端点"""
    return APIResponse(
        message="API服务正常运行",
        data={"service": "RESTful API演示", "status": "healthy"}
    )

# RESTful API设计原则2: HTTP方法使用
# 使用标准HTTP方法表示操作：POST创建资源，GET获取资源，PUT更新资源，DELETE删除资源
# RESTful API设计原则3: 状态管理与无状态性
# 无状态认证机制，每个请求携带完整认证信息，不依赖服务器会话状态
@app.post("/api/v1/users", response_model=UserResponse, status_code=201, tags=["用户"])
async def create_user(user: UserCreate = Body(...)):
    """创建新用户"""
    # 检查用户名唯一性
    for existing_user in users_db.values():
        if existing_user["username"] == user.username:
            raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 创建用户
    user_id = uuid.uuid4()
    new_user = {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "role": UserRole.user,
        "created_at": datetime.utcnow()
    }
    
    users_db[user_id] = new_user
    logger.info(f"用户创建成功: {user.username}")
    
    return UserResponse(**new_user)

@app.get("/api/v1/users", tags=["用户"])
async def list_users():
    """获取用户列表"""
    users = [UserResponse(**user) for user in users_db.values()]
    return APIResponse(data=users, message="用户列表获取成功")

@app.get("/api/v1/users/{user_id}", tags=["用户"])
async def get_user(user_id: UUID = Path(..., description="用户ID")):
    """获取特定用户信息"""
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return APIResponse(data=UserResponse(**user))

@app.put("/api/v1/users/{user_id}", tags=["用户"])
async def update_user(
    user_id: UUID = Path(..., description="用户ID"),
    user_data: dict = Body(..., description="更新数据")
):
    """更新用户信息"""
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 更新用户数据
    user.update(user_data)
    user["updated_at"] = datetime.utcnow()
    
    logger.info(f"用户更新成功: {user['username']}")
    return APIResponse(data=UserResponse(**user), message="用户更新成功")

@app.delete("/api/v1/users/{user_id}", tags=["用户"])
async def delete_user(
    user_id: UUID = Path(..., description="用户ID"),
    current_user: dict = Depends(require_role(UserRole.admin))
):
    """删除用户（管理员权限）"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    deleted_user = users_db.pop(user_id)
    logger.info(f"用户删除成功: {deleted_user['username']}")
    
    return APIResponse(message=f"用户 {deleted_user['username']} 删除成功")

# RESTful API设计原则8: 性能优化
# 速率限制保护API免受滥用，减少不必要的服务器负载
@app.get("/api/v1/stats", tags=["系统"])
async def get_stats():
    """获取系统统计信息"""
    return APIResponse(data={
        "total_users": len(users_db),
        "timestamp": datetime.utcnow()
    })

# RESTful API设计原则4: 错误处理
# 完善的错误处理机制，返回标准错误代码和清晰错误消息
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """全局HTTP异常处理"""
    logger.error(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return ErrorResponse(
        message=exc.detail,
        error_code=f"HTTP_{exc.status_code}"
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未预期错误: {str(exc)}")
    return ErrorResponse(
        message="服务器内部错误",
        error_code="INTERNAL_ERROR"
    )

# 主函数
if __name__ == "__main__":
    import uvicorn
    logger.info("正在启动RESTful API设计原则演示服务器...")
    uvicorn.run("restful_api设计原则演示:app", host="0.0.0.0", port=8000, reload=True)