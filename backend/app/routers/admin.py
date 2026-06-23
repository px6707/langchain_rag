from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_admin
from app.database import get_db
from app.models import User
from app.schemas import UserCreateRequest, UserListResponse, UserResponse, UserUpdateRequest
from app.services.user_service import UserService

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/users", response_model=UserListResponse)
async def list_users(db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    items, total = await service.list_users()
    return UserListResponse(
        items=[UserResponse.model_validate(u) for u in items],
        total=total,
    )


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(request: UserCreateRequest, db: AsyncSession = Depends(get_db)):
    service = UserService(db)
    user = await service.create_user(
        username=request.username,
        password=request.password,
        is_admin=request.is_admin,
    )
    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    request: UserUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    service = UserService(db)
    user = await service.update_user(
        user_id,
        is_active=request.is_active,
        is_admin=request.is_admin,
        password=request.password,
        current_user=current_user,
    )
    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    service = UserService(db)
    await service.delete_user(user_id, current_user)
    return {"message": "用户已删除"}
