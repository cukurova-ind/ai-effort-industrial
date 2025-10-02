import logging
from django.contrib.sessions.models import Session
from django.contrib.auth import logout
from django.shortcuts import redirect
from .models import LoggedInUser

logger = logging.getLogger(__name__)

class OneSessionPerUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            session_key = request.session.session_key
            
            try:
                logged_in_user = request.user.logged_in_user
                stored_session_key = logged_in_user.session_key

                # If this session doesn't match the stored one, logout current session
                if stored_session_key and stored_session_key != session_key:
                    # Check if the stored session still exists
                    if Session.objects.filter(session_key=stored_session_key).exists():
                        # Another session is active, logout this one
                        logger.warning(f"Multiple sessions detected for user {request.user.username}, logging out current session")
                        logout(request)
                        return redirect('/login/')
                    else:
                        # Stored session is stale, update with current session
                        logger.info(f"Updating stale session for user {request.user.username}")
                        logged_in_user.session_key = session_key
                        logged_in_user.save()
                elif not stored_session_key:
                    # No session key stored, set current one
                    logger.info(f"Setting session key for user {request.user.username}")
                    logged_in_user.session_key = session_key
                    logged_in_user.save()
                    
            except LoggedInUser.DoesNotExist:
                # This should be handled by the login view, but create if missing
                logger.info(f"Creating missing LoggedInUser record for {request.user.username}")
                LoggedInUser.objects.create(
                    user=request.user,
                    session_key=session_key,
                )

        response = self.get_response(request)
        return response