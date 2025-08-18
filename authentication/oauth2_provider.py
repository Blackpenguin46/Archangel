"""
OAuth2 Provider for agent-to-coordination communication authentication.

This module provides OAuth2 authentication services with:
- Authorization code flow
- Client credentials flow
- JWT token generation and validation
- Scope-based access control
"""

import json
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import base64


class GrantType(Enum):
    """OAuth2 grant types."""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    IMPLICIT = "implicit"


class TokenType(Enum):
    """OAuth2 token types."""
    BEARER = "Bearer"
    MAC = "MAC"


@dataclass
class OAuth2Client:
    """OAuth2 client registration."""
    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: List[str]
    grant_types: List[GrantType]
    scopes: Set[str]
    is_confidential: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    def verify_secret(self, provided_secret: str) -> bool:
        """Verify client secret."""
        return secrets.compare_digest(self.client_secret, provided_secret)
    
    def supports_grant_type(self, grant_type: GrantType) -> bool:
        """Check if client supports the grant type."""
        return grant_type in self.grant_types
    
    def has_scope(self, scope: str) -> bool:
        """Check if client has access to scope."""
        return scope in self.scopes


@dataclass
class AuthorizationCode:
    """OAuth2 authorization code."""
    code: str
    client_id: str
    redirect_uri: str
    scopes: Set[str]
    user_id: Optional[str] = None
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=10))
    used: bool = False
    
    def is_expired(self) -> bool:
        """Check if authorization code has expired."""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if authorization code is valid."""
        return not self.used and not self.is_expired()


@dataclass
class AccessToken:
    """OAuth2 access token."""
    token: str
    client_id: str
    scopes: Set[str]
    token_type: TokenType = TokenType.BEARER
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    user_id: Optional[str] = None
    refresh_token: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if access token has expired."""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if access token is valid."""
        return not self.is_expired()
    
    def has_scope(self, scope: str) -> bool:
        """Check if token has required scope."""
        return scope in self.scopes


@dataclass
class RefreshToken:
    """OAuth2 refresh token."""
    token: str
    client_id: str
    scopes: Set[str]
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    user_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if refresh token has expired."""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if refresh token is valid."""
        return not self.is_expired()


class OAuth2Provider:
    """
    OAuth2 Provider for agent authentication.
    
    Provides OAuth2 authentication services including:
    - Client registration and management
    - Authorization code flow
    - Client credentials flow
    - JWT token generation and validation
    """
    
    def __init__(self, issuer: str = "https://auth.archangel.local",
                 jwt_secret: Optional[str] = None):
        self.issuer = issuer
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.clients: Dict[str, OAuth2Client] = {}
        self.authorization_codes: Dict[str, AuthorizationCode] = {}
        self.access_tokens: Dict[str, AccessToken] = {}
        self.refresh_tokens: Dict[str, RefreshToken] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize default clients for agents
        self._initialize_default_clients()
    
    def _initialize_default_clients(self) -> None:
        """Initialize default OAuth2 clients for agent communication."""
        
        # Red Team agents client
        red_team_client = OAuth2Client(
            client_id="red_team_agents",
            client_secret=secrets.token_urlsafe(32),
            client_name="Red Team Agents",
            redirect_uris=["http://localhost:8080/callback"],
            grant_types=[GrantType.CLIENT_CREDENTIALS, GrantType.AUTHORIZATION_CODE],
            scopes={"agent:read", "agent:write", "coordination:participate", "intelligence:share"}
        )
        self.clients[red_team_client.client_id] = red_team_client
        
        # Blue Team agents client
        blue_team_client = OAuth2Client(
            client_id="blue_team_agents",
            client_secret=secrets.token_urlsafe(32),
            client_name="Blue Team Agents",
            redirect_uris=["http://localhost:8080/callback"],
            grant_types=[GrantType.CLIENT_CREDENTIALS, GrantType.AUTHORIZATION_CODE],
            scopes={"agent:read", "agent:write", "coordination:participate", "defense:manage", "monitoring:access"}
        )
        self.clients[blue_team_client.client_id] = blue_team_client
        
        # Coordinator client
        coordinator_client = OAuth2Client(
            client_id="archangel_coordinator",
            client_secret=secrets.token_urlsafe(32),
            client_name="Archangel Coordinator",
            redirect_uris=["http://localhost:8080/callback"],
            grant_types=[GrantType.CLIENT_CREDENTIALS],
            scopes={"coordination:manage", "agent:manage", "system:admin", "monitoring:admin"}
        )
        self.clients[coordinator_client.client_id] = coordinator_client
        
        # Monitoring system client
        monitoring_client = OAuth2Client(
            client_id="monitoring_system",
            client_secret=secrets.token_urlsafe(32),
            client_name="Monitoring System",
            redirect_uris=[],
            grant_types=[GrantType.CLIENT_CREDENTIALS],
            scopes={"monitoring:read", "metrics:collect", "logs:read"}
        )
        self.clients[monitoring_client.client_id] = monitoring_client
        
        self.logger.info(f"Initialized {len(self.clients)} default OAuth2 clients")
    
    def register_client(self, client_name: str, redirect_uris: List[str],
                       grant_types: List[GrantType], scopes: Set[str],
                       is_confidential: bool = True) -> OAuth2Client:
        """
        Register a new OAuth2 client.
        
        Args:
            client_name: Human-readable client name
            redirect_uris: List of valid redirect URIs
            grant_types: Supported grant types
            scopes: Allowed scopes for this client
            is_confidential: Whether client can securely store secrets
            
        Returns:
            Registered OAuth2 client
        """
        
        client_id = f"client_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32) if is_confidential else ""
        
        client = OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            client_name=client_name,
            redirect_uris=redirect_uris,
            grant_types=grant_types,
            scopes=scopes,
            is_confidential=is_confidential
        )
        
        self.clients[client_id] = client
        
        self.logger.info(f"Registered OAuth2 client: {client_name} (ID: {client_id})")
        return client
    
    def create_authorization_code(self, client_id: str, redirect_uri: str,
                                 scopes: Set[str], user_id: Optional[str] = None) -> Optional[str]:
        """
        Create authorization code for authorization code flow.
        
        Args:
            client_id: OAuth2 client ID
            redirect_uri: Redirect URI for callback
            scopes: Requested scopes
            user_id: User ID (if applicable)
            
        Returns:
            Authorization code or None if invalid request
        """
        
        # Validate client
        if client_id not in self.clients:
            self.logger.warning(f"Unknown client ID: {client_id}")
            return None
        
        client = self.clients[client_id]
        
        # Validate redirect URI
        if redirect_uri not in client.redirect_uris:
            self.logger.warning(f"Invalid redirect URI for client {client_id}: {redirect_uri}")
            return None
        
        # Validate grant type
        if not client.supports_grant_type(GrantType.AUTHORIZATION_CODE):
            self.logger.warning(f"Client {client_id} does not support authorization code flow")
            return None
        
        # Validate scopes
        invalid_scopes = scopes - client.scopes
        if invalid_scopes:
            self.logger.warning(f"Client {client_id} requested invalid scopes: {invalid_scopes}")
            return None
        
        # Generate authorization code
        code = secrets.token_urlsafe(32)
        
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            user_id=user_id
        )
        
        self.authorization_codes[code] = auth_code
        
        self.logger.info(f"Created authorization code for client {client_id}")
        return code
    
    def exchange_authorization_code(self, client_id: str, client_secret: str,
                                  code: str, redirect_uri: str) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            code: Authorization code
            redirect_uri: Redirect URI used in authorization request
            
        Returns:
            Token response or None if invalid
        """
        
        # Validate client
        if client_id not in self.clients:
            self.logger.warning(f"Unknown client ID: {client_id}")
            return None
        
        client = self.clients[client_id]
        
        # Verify client secret
        if client.is_confidential and not client.verify_secret(client_secret):
            self.logger.warning(f"Invalid client secret for {client_id}")
            return None
        
        # Validate authorization code
        if code not in self.authorization_codes:
            self.logger.warning(f"Unknown authorization code: {code}")
            return None
        
        auth_code = self.authorization_codes[code]
        
        # Verify authorization code
        if not auth_code.is_valid():
            self.logger.warning(f"Invalid or expired authorization code: {code}")
            del self.authorization_codes[code]
            return None
        
        if auth_code.client_id != client_id:
            self.logger.warning(f"Authorization code client mismatch: expected {auth_code.client_id}, got {client_id}")
            return None
        
        if auth_code.redirect_uri != redirect_uri:
            self.logger.warning(f"Redirect URI mismatch for authorization code")
            return None
        
        # Mark code as used
        auth_code.used = True
        
        # Generate tokens
        access_token = self._generate_access_token(client_id, auth_code.scopes, auth_code.user_id)
        refresh_token = self._generate_refresh_token(client_id, auth_code.scopes, auth_code.user_id)
        
        # Clean up authorization code
        del self.authorization_codes[code]
        
        client.last_used = datetime.now()
        
        return {
            "access_token": access_token.token,
            "token_type": access_token.token_type.value,
            "expires_in": int((access_token.expires_at - datetime.now()).total_seconds()),
            "refresh_token": refresh_token.token,
            "scope": " ".join(access_token.scopes)
        }
    
    def client_credentials_grant(self, client_id: str, client_secret: str,
                               scopes: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Handle client credentials grant flow.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            scopes: Requested scopes (optional)
            
        Returns:
            Token response or None if invalid
        """
        
        # Validate client
        if client_id not in self.clients:
            self.logger.warning(f"Unknown client ID: {client_id}")
            return None
        
        client = self.clients[client_id]
        
        # Verify client secret
        if not client.verify_secret(client_secret):
            self.logger.warning(f"Invalid client secret for {client_id}")
            return None
        
        # Validate grant type
        if not client.supports_grant_type(GrantType.CLIENT_CREDENTIALS):
            self.logger.warning(f"Client {client_id} does not support client credentials flow")
            return None
        
        # Use client's default scopes if none requested
        if scopes is None:
            scopes = client.scopes
        else:
            # Validate requested scopes
            invalid_scopes = scopes - client.scopes
            if invalid_scopes:
                self.logger.warning(f"Client {client_id} requested invalid scopes: {invalid_scopes}")
                return None
        
        # Generate access token
        access_token = self._generate_access_token(client_id, scopes)
        
        client.last_used = datetime.now()
        
        return {
            "access_token": access_token.token,
            "token_type": access_token.token_type.value,
            "expires_in": int((access_token.expires_at - datetime.now()).total_seconds()),
            "scope": " ".join(access_token.scopes)
        }
    
    def refresh_access_token(self, refresh_token: str, client_id: str,
                           client_secret: str) -> Optional[Dict[str, Any]]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            
        Returns:
            New token response or None if invalid
        """
        
        # Validate client
        if client_id not in self.clients:
            self.logger.warning(f"Unknown client ID: {client_id}")
            return None
        
        client = self.clients[client_id]
        
        # Verify client secret
        if client.is_confidential and not client.verify_secret(client_secret):
            self.logger.warning(f"Invalid client secret for {client_id}")
            return None
        
        # Validate refresh token
        if refresh_token not in self.refresh_tokens:
            self.logger.warning(f"Unknown refresh token")
            return None
        
        refresh_token_obj = self.refresh_tokens[refresh_token]
        
        if not refresh_token_obj.is_valid():
            self.logger.warning(f"Invalid or expired refresh token")
            del self.refresh_tokens[refresh_token]
            return None
        
        if refresh_token_obj.client_id != client_id:
            self.logger.warning(f"Refresh token client mismatch")
            return None
        
        # Generate new tokens
        new_access_token = self._generate_access_token(
            client_id, refresh_token_obj.scopes, refresh_token_obj.user_id
        )
        new_refresh_token = self._generate_refresh_token(
            client_id, refresh_token_obj.scopes, refresh_token_obj.user_id
        )
        
        # Revoke old refresh token
        del self.refresh_tokens[refresh_token]
        
        client.last_used = datetime.now()
        
        return {
            "access_token": new_access_token.token,
            "token_type": new_access_token.token_type.value,
            "expires_in": int((new_access_token.expires_at - datetime.now()).total_seconds()),
            "refresh_token": new_refresh_token.token,
            "scope": " ".join(new_access_token.scopes)
        }
    
    def _generate_access_token(self, client_id: str, scopes: Set[str],
                              user_id: Optional[str] = None) -> AccessToken:
        """Generate JWT access token."""
        
        now = datetime.now()
        expires_at = now + timedelta(hours=1)
        
        # JWT payload
        payload = {
            "iss": self.issuer,
            "sub": user_id or client_id,
            "aud": "archangel-agents",
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "client_id": client_id,
            "scope": " ".join(scopes)
        }
        
        # Generate JWT token
        if JWT_AVAILABLE:
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        else:
            # Mock JWT token
            import base64
            import json
            mock_payload = json.dumps(payload)
            token = f"mock_jwt_{base64.b64encode(mock_payload.encode()).decode()}"
        
        access_token = AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            user_id=user_id
        )
        
        self.access_tokens[token] = access_token
        
        return access_token
    
    def _generate_refresh_token(self, client_id: str, scopes: Set[str],
                               user_id: Optional[str] = None) -> RefreshToken:
        """Generate refresh token."""
        
        token = secrets.token_urlsafe(32)
        
        refresh_token = RefreshToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            user_id=user_id
        )
        
        self.refresh_tokens[token] = refresh_token
        
        return refresh_token
    
    def validate_access_token(self, token: str, required_scope: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Validate access token and return token info.
        
        Args:
            token: Access token to validate
            required_scope: Required scope for access
            
        Returns:
            Token information or None if invalid
        """
        
        try:
            # Decode JWT token
            if JWT_AVAILABLE:
                payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            else:
                # Mock JWT token validation
                if token.startswith("mock_jwt_"):
                    import base64
                    import json
                    encoded_payload = token[9:]  # Remove "mock_jwt_" prefix
                    mock_payload = base64.b64decode(encoded_payload).decode()
                    payload = json.loads(mock_payload)
                else:
                    raise ValueError("Invalid mock JWT token")
            
            # Check if token exists in our store
            if token not in self.access_tokens:
                self.logger.warning("Token not found in token store")
                return None
            
            access_token = self.access_tokens[token]
            
            # Check if token is still valid
            if not access_token.is_valid():
                self.logger.warning("Access token has expired")
                del self.access_tokens[token]
                return None
            
            # Check required scope
            if required_scope and not access_token.has_scope(required_scope):
                self.logger.warning(f"Token does not have required scope: {required_scope}")
                return None
            
            return {
                "client_id": access_token.client_id,
                "user_id": access_token.user_id,
                "scopes": list(access_token.scopes),
                "expires_at": access_token.expires_at.isoformat(),
                "token_type": access_token.token_type.value
            }
            
        except Exception as e:
            if JWT_AVAILABLE:
                if "ExpiredSignatureError" in str(type(e)):
                    self.logger.warning("JWT token has expired")
                else:
                    self.logger.warning(f"Invalid JWT token: {e}")
            else:
                self.logger.warning(f"Mock JWT validation error: {e}")
            return None
    
    def revoke_token(self, token: str, client_id: str, client_secret: str) -> bool:
        """
        Revoke access or refresh token.
        
        Args:
            token: Token to revoke
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            
        Returns:
            True if token was revoked successfully
        """
        
        # Validate client
        if client_id not in self.clients:
            return False
        
        client = self.clients[client_id]
        
        # Verify client secret
        if client.is_confidential and not client.verify_secret(client_secret):
            return False
        
        # Try to revoke access token
        if token in self.access_tokens:
            access_token = self.access_tokens[token]
            if access_token.client_id == client_id:
                del self.access_tokens[token]
                self.logger.info(f"Revoked access token for client {client_id}")
                return True
        
        # Try to revoke refresh token
        if token in self.refresh_tokens:
            refresh_token = self.refresh_tokens[token]
            if refresh_token.client_id == client_id:
                del self.refresh_tokens[token]
                self.logger.info(f"Revoked refresh token for client {client_id}")
                return True
        
        return False
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an OAuth2 client."""
        
        if client_id not in self.clients:
            return None
        
        client = self.clients[client_id]
        
        return {
            "client_id": client.client_id,
            "client_name": client.client_name,
            "redirect_uris": client.redirect_uris,
            "grant_types": [gt.value for gt in client.grant_types],
            "scopes": list(client.scopes),
            "is_confidential": client.is_confidential,
            "created_at": client.created_at.isoformat(),
            "last_used": client.last_used.isoformat() if client.last_used else None
        }
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get OAuth2 server statistics."""
        
        active_access_tokens = len([token for token in self.access_tokens.values() if token.is_valid()])
        active_refresh_tokens = len([token for token in self.refresh_tokens.values() if token.is_valid()])
        
        return {
            "issuer": self.issuer,
            "total_clients": len(self.clients),
            "total_access_tokens": len(self.access_tokens),
            "active_access_tokens": active_access_tokens,
            "total_refresh_tokens": len(self.refresh_tokens),
            "active_refresh_tokens": active_refresh_tokens,
            "pending_authorization_codes": len([code for code in self.authorization_codes.values() if code.is_valid()])
        }