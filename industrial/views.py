import os
import logging
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.sessions.models import Session
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme
from engine.models import LoggedInUser
from engine.utils.user_folders import ensure_user_folders
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm

logger = logging.getLogger(__name__)


def custom_login_view(request):
    next_url = request.GET.get('next') or request.POST.get('next')
    form = AuthenticationForm(request, data=request.POST or None)
    logout_flag = False
    
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        force_login = request.POST.get("force_login") == "on"
        user = authenticate(request, username=username, password=password)

        if user:
            logger.info(f"User {user.username} attempting login")
            
            try:
                logged_in = LoggedInUser.objects.get(user=user)
                session_exists = Session.objects.filter(session_key=logged_in.session_key).exists()
                
                if session_exists and not force_login:
                    logger.warning(f"User {user.username} already logged in from another session")
                    messages.error(request, "You are already logged in from another device or browser.")
                    logout_flag = True
                    return render(request, "login.html", {"form": form, "next": next_url, "logout": logout_flag})
                elif force_login and session_exists:
                    # Force logout from other session
                    logger.info(f"Force logout for user {user.username} - terminating other session")
                    Session.objects.filter(session_key=logged_in.session_key).delete()
                    logged_in.delete()
                elif not session_exists:
                    # Stale session record, clean it up
                    logger.info(f"Cleaning stale session record for user {user.username}")
                    logged_in.delete()
            except LoggedInUser.DoesNotExist:
                logger.info(f"No existing login record for user {user.username}")
                pass

            # Perform login
            login(request, user)
            request.session.save()
            
            # Create/update logged in user record
            LoggedInUser.objects.update_or_create(
                user=user, 
                defaults={"session_key": request.session.session_key}
            )
            logger.info(f"User {user.username} successfully logged in with session {request.session.session_key}")
            
            # Ensure user folders exist
            if ensure_user_folders(user):
                logger.info(f"User folders verified/created for {user.username}")
            else:
                logger.error(f"Failed to create user folders for {user.username}")

            if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
                return redirect(next_url)
            return redirect("/")

        else:
            logger.warning(f"Failed login attempt for username: {username}")
            messages.error(request, "Invalid credentials")
            return render(request, "login.html", {"form": form, "next": next_url, "logout": logout_flag})
        
    return render(request, "login.html", {"form": form, "next": next_url, "logout": logout_flag})


def logout_all(request):
    username = request.user.username if request.user.is_authenticated else "Anonymous"
    logger.info(f"Logout request for user: {username}")
    
    try:
        if request.user.is_authenticated:
            LoggedInUser.objects.filter(user=request.user).delete()
            logger.info(f"Cleared login record for user: {username}")
    except Exception as e:
        logger.error(f"Error clearing login record: {e}")
        
    logout(request)
    return redirect('/login/')


def main(req):
    return render(req, "main_page.html", {"exp":None})