This architecture cleanly separates concerns: **Agentic workflows handle business logic**, while the **HIL orchestrator handles routing between AI and humans**. Perfect! 🎯

---

## 🔐 API Token Management & Security Policies

### The Problem

When building integrations with external services (Shopify, Gmail, Slack, etc.), we face a critical security dilemma:

```
❌ INSECURE APPROACH:
User → Gives API Token → Your System → Stores Token → Uses Token
                         (Single Point of Failure)
                         (You're responsible for breach)
                         (User must trust you completely)

Problems:
1. You store sensitive credentials
2. If your DB is breached, all tokens exposed
3. Users must fully trust your security
4. Compliance nightmare (SOC 2, ISO 27001)
5. Token revocation requires user action
```

### The Solution: OAuth 2.0 with Token Delegation

```
✅ SECURE APPROACH (OAuth 2.0):
User → OAuth Flow → Service (Shopify) → Temporary Token → Your System
       (No password)   (User authorizes)  (Limited scope)   (Encrypted)

Benefits:
1. You never see user's password
2. Tokens are scoped (limited permissions)
3. Tokens are temporary (auto-expire)
4. User can revoke anytime
5. Industry standard (trusted)
```

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Token Security Architecture                │
└──────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │   User      │
                    └──────┬──────┘
                           │ 1. "Connect Shopify"
                           ▼
                    ┌─────────────┐
                    │ HIL System  │
                    │ (Your API)  │
                    └──────┬──────┘
                           │ 2. Redirect to OAuth
                           ▼
                    ┌─────────────┐
                    │  Shopify    │
                    │  (OAuth)    │
                    └──────┬──────┘
                           │ 3. User authorizes
                           ▼
                    ┌─────────────┐
                    │  Shopify    │
                    │  (Callback) │
                    └──────┬──────┘
                           │ 4. Authorization code
                           ▼
                    ┌─────────────┐
                    │ HIL System  │
                    │ (Exchange)  │
                    └──────┬──────┘
                           │ 5. Exchange for token
                           ▼
                    ┌─────────────┐
                    │  Shopify    │
                    │  (Returns)  │
                    └──────┬──────┘
                           │ 6. Access Token + Refresh Token
                           ▼
                    ┌─────────────┐
                    │ HIL System  │
                    │ (Encrypt &  │
                    │  Store)     │
                    └──────┬──────┘
                           │ 7. Use encrypted token
                           ▼
                    ┌─────────────┐
                    │  Shopify    │
                    │  API Calls  │
                    └─────────────┘
```

### Token Storage Strategy

```python
# app/security/token_manager.py

from cryptography.fernet import Fernet
from typing import Optional, Dict, Any
import base64
import hashlib
import secrets

class TokenManager:
    """
    Secure token management with encryption, rotation, and auditing.
    
    Features:
    - AES-256 encryption at rest
    - Per-entity encryption keys (data isolation)
    - Automatic token refresh
    - Audit logging
    - Token expiration handling
    """
    
    def __init__(self, master_key: bytes, db_pool, redis_client):
        """
        master_key: Master encryption key (from KMS/Vault)
        """
        self.master_key = master_key
        self.db = db_pool
        self.redis = redis_client
    
    def _derive_entity_key(self, entity_id: str, salt: bytes) -> bytes:
        """
        Derive encryption key per entity.
        Even if one entity key is compromised, others remain safe.
        """
        kdf_input = self.master_key + entity_id.encode() + salt
        return hashlib.pbkdf2_hmac('sha256', kdf_input, salt, 100000)
    
    async def store_token(
        self,
        entity_id: str,
        service: str,
        token_data: Dict[str, Any]
    ) -> str:
        """
        Store token with encryption.
        
        Args:
            entity_id: User/customer ID
            service: Service name (shopify, gmail, etc.)
            token_data: {
                access_token: str,
                refresh_token: str (optional),
                expires_at: int (timestamp),
                scope: str
            }
        
        Returns:
            token_id: Identifier for stored token
        """
        
        # Generate unique salt per token
        salt = secrets.token_bytes(32)
        
        # Derive entity-specific key
        entity_key = self._derive_entity_key(entity_id, salt)
        fernet = Fernet(base64.urlsafe_b64encode(entity_key))
        
        # Encrypt token data
        encrypted_access = fernet.encrypt(token_data['access_token'].encode())
        encrypted_refresh = None
        if token_data.get('refresh_token'):
            encrypted_refresh = fernet.encrypt(token_data['refresh_token'].encode())
        
        # Store in database
        token_id = secrets.token_urlsafe(32)
        
        await self.db.execute("""
            INSERT INTO oauth_tokens
            (id, entity_id, service, encrypted_access_token, encrypted_refresh_token,
             salt, expires_at, scope, created_at, last_used_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
        """, token_id, entity_id, service, encrypted_access, encrypted_refresh,
            salt, token_data.get('expires_at'), token_data.get('scope'))
        
        # Audit log
        await self._audit_log(
            entity_id=entity_id,
            action="token_stored",
            service=service,
            metadata={"token_id": token_id}
        )
        
        logger.info(
            "token_stored",
            entity_id=entity_id,
            service=service,
            token_id=token_id
        )
        
        return token_id
    
    async def get_token(
        self,
        entity_id: str,
        service: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt token.
        Automatically refreshes if expired.
        """
        
        # Get encrypted token
        row = await self.db.fetchrow("""
            SELECT *
            FROM oauth_tokens
            WHERE entity_id = $1 AND service = $2
            AND revoked_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, entity_id, service)
        
        if not row:
            return None
        
        # Check if expired
        if row['expires_at'] and row['expires_at'] < time.time():
            # Try to refresh
            if row['encrypted_refresh_token']:
                refreshed = await self._refresh_token(entity_id, service, dict(row))
                if refreshed:
                    return await self.get_token(entity_id, service)  # Recursive call
            
            # Cannot refresh - token is expired
            return None
        
        # Derive key and decrypt
        entity_key = self._derive_entity_key(entity_id, row['salt'])
        fernet = Fernet(base64.urlsafe_b64encode(entity_key))
        
        access_token = fernet.decrypt(row['encrypted_access_token']).decode()
        refresh_token = None
        if row['encrypted_refresh_token']:
            refresh_token = fernet.decrypt(row['encrypted_refresh_token']).decode()
        
        # Update last used
        await self.db.execute("""
            UPDATE oauth_tokens
            SET last_used_at = NOW()
            WHERE id = $1
        """, row['id'])
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': row['expires_at'],
            'scope': row['scope']
        }
    
    async def _refresh_token(
        self,
        entity_id: str,
        service: str,
        old_token_row: Dict[str, Any]
    ) -> bool:
        """
        Refresh expired token using refresh_token.
        """
        
        # Decrypt refresh token
        entity_key = self._derive_entity_key(entity_id, old_token_row['salt'])
        fernet = Fernet(base64.urlsafe_b64encode(entity_key))
        refresh_token = fernet.decrypt(old_token_row['encrypted_refresh_token']).decode()
        
        # Call service's refresh endpoint
        try:
            new_tokens = await self._call_refresh_endpoint(service, refresh_token)
            
            # Store new token
            await self.store_token(
                entity_id=entity_id,
                service=service,
                token_data=new_tokens
            )
            
            # Revoke old token
            await self.db.execute("""
                UPDATE oauth_tokens
                SET revoked_at = NOW()
                WHERE id = $1
            """, old_token_row['id'])
            
            logger.info(
                "token_refreshed",
                entity_id=entity_id,
                service=service
            )
            
            return True
        
        except Exception as e:
            logger.error(
                "token_refresh_failed",
                entity_id=entity_id,
                service=service,
                error=str(e)
            )
            return False
    
    async def revoke_token(
        self,
        entity_id: str,
        service: str,
        reason: str = "user_requested"
    ):
        """
        Revoke token (user disconnect).
        """
        
        # Mark as revoked in DB
        await self.db.execute("""
            UPDATE oauth_tokens
            SET revoked_at = NOW(), revocation_reason = $1
            WHERE entity_id = $2 AND service = $3
            AND revoked_at IS NULL
        """, reason, entity_id, service)
        
        # Audit log
        await self._audit_log(
            entity_id=entity_id,
            action="token_revoked",
            service=service,
            metadata={"reason": reason}
        )
        
        logger.info(
            "token_revoked",
            entity_id=entity_id,
            service=service,
            reason=reason
        )
    
    async def rotate_master_key(self, new_master_key: bytes):
        """
        Rotate master encryption key (re-encrypt all tokens).
        Run as background job during maintenance window.
        """
        
        # Get all active tokens
        tokens = await self.db.fetch("""
            SELECT *
            FROM oauth_tokens
            WHERE revoked_at IS NULL
        """)
        
        logger.info("master_key_rotation_started", total_tokens=len(tokens))
        
        for token in tokens:
            try:
                # Decrypt with old key
                old_entity_key = self._derive_entity_key(
                    token['entity_id'],
                    token['salt']
                )
                old_fernet = Fernet(base64.urlsafe_b64encode(old_entity_key))
                access_token = old_fernet.decrypt(token['encrypted_access_token']).decode()
                
                refresh_token = None
                if token['encrypted_refresh_token']:
                    refresh_token = old_fernet.decrypt(token['encrypted_refresh_token']).decode()
                
                # Generate new salt
                new_salt = secrets.token_bytes(32)
                
                # Encrypt with new key
                self.master_key = new_master_key
                new_entity_key = self._derive_entity_key(
                    token['entity_id'],
                    new_salt
                )
                new_fernet = Fernet(base64.urlsafe_b64encode(new_entity_key))
                
                new_encrypted_access = new_fernet.encrypt(access_token.encode())
                new_encrypted_refresh = None
                if refresh_token:
                    new_encrypted_refresh = new_fernet.encrypt(refresh_token.encode())
                
                # Update database
                await self.db.execute("""
                    UPDATE oauth_tokens
                    SET 
                        encrypted_access_token = $1,
                        encrypted_refresh_token = $2,
                        salt = $3
                    WHERE id = $4
                """, new_encrypted_access, new_encrypted_refresh, new_salt, token['id'])
            
            except Exception as e:
                logger.error(
                    "token_rotation_failed",
                    token_id=token['id'],
                    error=str(e)
                )
        
        logger.info("master_key_rotation_completed")
    
    async def _audit_log(
        self,
        entity_id: str,
        action: str,
        service: str,
        metadata: Dict[str, Any]
    ):
        """Record audit log for compliance"""
        
        await self.db.execute("""
            INSERT INTO token_audit_log
            (entity_id, action, service, metadata, ip_address, user_agent, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
        """, entity_id, action, service, json.dumps(metadata),
            "system", "hil-system")  # TODO: Get actual IP/UA
    
    async def _call_refresh_endpoint(
        self,
        service: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """
        Call service-specific refresh endpoint.
        Each service has different OAuth implementation.
        """
        
        # Service-specific refresh logic
        if service == "shopify":
            # Shopify OAuth refresh
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://shopify.com/admin/oauth/access_token",
                    json={
                        "client_id": os.getenv("SHOPIFY_CLIENT_ID"),
                        "client_secret": os.getenv("SHOPIFY_CLIENT_SECRET"),
                        "refresh_token": refresh_token
                    }
                )
                data = response.json()
                return {
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token"),
                    "expires_at": int(time.time()) + data["expires_in"],
                    "scope": data["scope"]
                }
        
        elif service == "gmail":
            # Google OAuth refresh
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token"
                    }
                )
                data = response.json()
                return {
                    "access_token": data["access_token"],
                    "refresh_token": refresh_token,  # Google doesn't return new refresh token
                    "expires_at": int(time.time()) + data["expires_in"],
                    "scope": data.get("scope")
                }
        
        else:
            raise ValueError(f"Unsupported service: {service}")


# Database schema for token storage
"""
CREATE TABLE oauth_tokens (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  service TEXT NOT NULL,
  encrypted_access_token BYTEA NOT NULL,
  encrypted_refresh_token BYTEA,
  salt BYTEA NOT NULL,
  expires_at BIGINT,  -- Unix timestamp
  scope TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_used_at TIMESTAMPTZ DEFAULT NOW(),
  revoked_at TIMESTAMPTZ,
  revocation_reason TEXT,
  
  CONSTRAINT unique_entity_service UNIQUE (entity_id, service) WHERE revoked_at IS NULL
);

CREATE INDEX idx_oauth_tokens_entity ON oauth_tokens(entity_id);
CREATE INDEX idx_oauth_tokens_expires ON oauth_tokens(expires_at) WHERE revoked_at IS NULL;

CREATE TABLE token_audit_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_id TEXT NOT NULL,
  action TEXT NOT NULL,  -- token_stored, token_retrieved, token_refreshed, token_revoked
  service TEXT NOT NULL,
  metadata JSONB,
  ip_address TEXT,
  user_agent TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_token_audit_entity ON token_audit_log(entity_id);
CREATE INDEX idx_token_audit_created ON token_audit_log(created_at DESC);
"""
```

### OAuth 2.0 Flow Implementation

```python
# app/api/oauth.py

from fastapi import APIRouter, HTTPException, Request
from typing import Optional

router = APIRouter(prefix="/api/v1/oauth", tags=["oauth"])

@router.post("/connect/{service}")
async def initiate_oauth(
    service: str,
    entity_id: str,
    redirect_uri: str,
    scopes: Optional[List[str]] = None
):
    """
    Step 1: Initiate OAuth flow.
    Returns authorization URL for user to visit.
    
    Example:
    POST /api/v1/oauth/connect/shopify
    {
      "entity_id": "customer_123",
      "redirect_uri": "https://yourapp.com/oauth/callback",
      "scopes": ["read_orders", "write_orders"]
    }
    """
    
    # Generate state parameter (CSRF protection)
    state = secrets.token_urlsafe(32)
    
    # Store state temporarily (5 min expiry)
    await redis.setex(
        f"oauth_state:{state}",
        300,
        json.dumps({
            "entity_id": entity_id,
            "service": service,
            "redirect_uri": redirect_uri
        })
    )
    
    # Build authorization URL based on service
    auth_url = build_auth_url(service, state, scopes, redirect_uri)
    
    return {
        "authorization_url": auth_url,
        "state": state,
        "expires_in": 300
    }


@router.get("/callback/{service}")
async def oauth_callback(
    service: str,
    code: str,
    state: str,
    request: Request,
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Step 2: OAuth callback handler.
    Service redirects here after user authorizes.
    
    Example callback from Shopify:
    GET /api/v1/oauth/callback/shopify?code=abc123&state=xyz789
    """
    
    # Verify state (CSRF protection)
    state_data = await redis.get(f"oauth_state:{state}")
    if not state_data:
        raise HTTPException(status_code=400, detail="Invalid or expired state")
    
    state_data = json.loads(state_data)
    entity_id = state_data["entity_id"]
    redirect_uri = state_data["redirect_uri"]
    
    # Delete state (one-time use)
    await redis.delete(f"oauth_state:{state}")
    
    try:
        # Exchange authorization code for tokens
        tokens = await exchange_code_for_tokens(service, code, redirect_uri)
        
        # Store encrypted tokens
        token_id = await token_manager.store_token(
            entity_id=entity_id,
            service=service,
            token_data=tokens
        )
        
        # Redirect back to application
        return RedirectResponse(
            url=f"{redirect_uri}?success=true&service={service}"
        )
    
    except Exception as e:
        logger.error(
            "oauth_exchange_failed",
            service=service,
            error=str(e),
            exc_info=True
        )
        
        return RedirectResponse(
            url=f"{redirect_uri}?success=false&error={str(e)}"
        )


@router.delete("/disconnect/{service}")
async def disconnect_service(
    service: str,
    entity_id: str,
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Disconnect service (revoke token).
    
    Example:
    DELETE /api/v1/oauth/disconnect/shopify?entity_id=customer_123
    """
    
    await token_manager.revoke_token(
        entity_id=entity_id,
        service=service,
        reason="user_requested"
    )
    
    return {"status": "disconnected", "service": service}


@router.get("/status/{service}")
async def get_connection_status(
    service: str,
    entity_id: str,
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Check if service is connected.
    
    Returns:
    {
      "connected": true,
      "service": "shopify",
      "scope": "read_orders,write_orders",
      "expires_at": 1234567890,
      "last_used": "2025-01-12T10:30:00Z"
    }
    """
    
    token = await token_manager.get_token(entity_id, service)
    
    if not token:
        return {
            "connected": False,
            "service": service
        }
    
    return {
        "connected": True,
        "service": service,
        "scope": token.get("scope"),
        "expires_at": token.get("expires_at"),
        "expires_in": token.get("expires_at") - int(time.time()) if token.get("expires_at") else None
    }


def build_auth_url(
    service: str,
    state: str,
    scopes: Optional[List[str]],
    redirect_uri: str
) -> str:
    """Build service-specific authorization URL"""
    
    if service == "shopify":
        shop = "your-shop.myshopify.com"  # TODO: Get from entity
        scope = ",".join(scopes or ["read_orders", "write_orders"])
        return (
            f"https://{shop}/admin/oauth/authorize?"
            f"client_id={os.getenv('SHOPIFY_CLIENT_ID')}&"
            f"scope={scope}&"
            f"redirect_uri={redirect_uri}&"
            f"state={state}"
        )
    
    elif service == "gmail":
        scope = " ".join(scopes or ["https://www.googleapis.com/auth/gmail.send"])
        return (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={os.getenv('GOOGLE_CLIENT_ID')}&"
            f"redirect_uri={redirect_uri}&"
            f"response_type=code&"
            f"scope={scope}&"
            f"state={state}&"
            f"access_type=offline&"  # Request refresh token
            f"prompt=consent"  # Force consent screen
        )
    
    elif service == "slack":
        scope = ",".join(scopes or ["chat:write", "channels:read"])
        return (
            f"https://slack.com/oauth/v2/authorize?"
            f"client_id={os.getenv('SLACK_CLIENT_ID')}&"
            f"scope={scope}&"
            f"redirect_uri={redirect_uri}&"
            f"state={state}"
        )
    
    else:
        raise ValueError(f"Unsupported service: {service}")


async def exchange_code_for_tokens(
    service: str,
    code: str,
    redirect_uri: str
) -> Dict[str, Any]:
    """Exchange authorization code for access token"""
    
    if service == "shopify":
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://your-shop.myshopify.com/admin/oauth/access_token",
                json={
                    "client_id": os.getenv("SHOPIFY_CLIENT_ID"),
                    "client_secret": os.getenv("SHOPIFY_CLIENT_SECRET"),
                    "code": code
                }
            )
            data = response.json()
            return {
                "access_token": data["access_token"],
                "scope": data["scope"],
                "expires_at": None  # Shopify tokens don't expire
            }
    
    elif service == "gmail":
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code"
                }
            )
            data = response.json()
            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "expires_at": int(time.time()) + data["expires_in"],
                "scope": data["scope"]
            }
    
    else:
        raise ValueError(f"Unsupported service: {service}")
```

### Token Usage in Tools

```python
# app/tools/secure_tool_executor.py

class SecureToolExecutor:
    """
    Execute tools with automatic token retrieval and refresh.
    """
    
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
    
    async def execute(
        self,
        entity_id: str,
        service: str,
        action: str,
        params: dict
    ) -> Any:
        """
        Execute tool with automatic token management.
        """
        
        # Get token (auto-refreshes if expired)
        token = await self.token_manager.get_token(entity_id, service)
        
        if not token:
            raise ValueError(
                f"No valid token for {service}. "
                f"Please connect your {service} account."
            )
        
        # Execute action with token
        try:
            if service == "shopify":
                return await self._execute_shopify(
                    action,
                    params,
                    token["access_token"]
                )
            
            elif service == "gmail":
                return await self._execute_gmail(
                    action,
                    params,
                    token["access_token"]
                )
            
            else:
                raise ValueError(f"Unsupported service: {service}")
        
        except UnauthorizedError:
            # Token might be invalid - try refresh
            await self.token_manager._refresh_token(entity_id, service, token)
            
            # Retry once
            token = await self.token_manager.get_token(entity_id, service)
            if not token:
                raise ValueError(f"Failed to refresh {service} token")
            
            return await self.execute(entity_id, service, action, params)
    
    async def _execute_shopify(
        self,
        action: str,
        params: dict,
        access_token: str
    ) -> dict:
        """Execute Shopify API call"""
        
        shop = params.get("shop", "your-shop.myshopify.com")
        
        async with httpx.AsyncClient() as client:
            if action == "get_order":
                response = await client.get(
                    f"https://{shop}/admin/api/2024-01/orders/{params['order_id']}.json",
                    headers={"X-Shopify-Access-Token": access_token}
                )
            
            elif action == "create_return":
                response = await client.post(
                    f"https://{shop}/admin/api/2024-01/returns.json",
                    headers={"X-Shopify-Access-Token": access_token},
                    json=params
                )
            
            else:
                raise ValueError(f"Unknown Shopify action: {action}")
            
            if response.status_code == 401:
                raise UnauthorizedError("Invalid token")
            
            response.raise_for_status()
            return response.json()
```

### Security Best Practices

```python
# config/security_policy.yaml

token_security:
  # Encryption
  encryption:
    algorithm: "AES-256-GCM"
    key_derivation: "PBKDF2-SHA256"
    iterations: 100000
    master_key_rotation: "quarterly"  # Every 3 months
  
  # Storage
  storage:
    per_entity_encryption: true  # Separate keys per user
    salt_length: 32  # bytes
    never_log_tokens: true
    never_expose_in_api: true
  
  # Access Control
  access:
    require_entity_ownership: true  # Can only access own tokens
    admin_cannot_decrypt: true  # Even admins can't see tokens
    audit_all_access: true
  
  # Token Lifecycle
  lifecycle:
    max_age: 90  # days (force re-auth)
    refresh_before_expiry: 300  # seconds (5 min)
    revoke_on_suspicious_activity: true
    auto_cleanup_revoked: 30  # days
  
  # Compliance
  compliance:
    gdpr_deletion: true  # Delete on user request
    retention_period: 90  # days after revocation
    audit_retention: 2555  # days (7 years)
    encrypt_audit_logs: true

# Master Key Management
master_key:
  source: "aws_kms"  # or "azure_key_vault", "gcp_kms", "hashicorp_vault"
  rotation_schedule: "quarterly"
  backup_keys: 2  # Keep 2 previous keys for decryption during rotation
  access_control:
    - role: "hil-system-production"
      permissions: ["decrypt", "encrypt"]
    - role: "hil-system-key-rotation"
      permissions: ["decrypt", "encrypt", "rotate"]

# Rate Limiting per Token
rate_limiting:
  per_token:
    requests_per_minute: 60
    requests_per_hour: 1000
    requests_per_day: 10000
  
  enforcement:
    strategy: "sliding_window"
    block_on_exceed: true
    notify_entity: true

# Monitoring & Alerts
monitoring:
  alerts:
    - condition: "token_refresh_failed"
      severity: "warning"
      action: "notify_entity"
    
    - condition: "token_used_from_new_ip"
      severity: "info"
      action: "log_only"
    
    - condition: "multiple_failed_refresh_attempts"
      severity: "critical"
      action: "revoke_token_and_notify"
    
    - condition: "master_key_rotation_failed"
      severity: "critical"
      action: "page_on_call"
```

