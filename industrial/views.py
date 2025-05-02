import os
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.sessions.models import Session
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme
from engine.models import LoggedInUser
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm


def custom_login_view(request):
    next_url = request.GET.get('next') or request.POST.get('next')
    form = AuthenticationForm(request, data=request.POST or None)
    logout = False
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        force_login = request.POST.get("force_login") == "on"
        user = authenticate(request, username=username, password=password)

        if user:
            try:
                logged_in = LoggedInUser.objects.get(user=user)
                session_exists = Session.objects.filter(session_key=logged_in.session_key).exists()
                if session_exists and not force_login:
                    messages.error(request, "You are already logged in from another device or browser.")
                    logout = True
                    return render(request, "login.html", {"form": form, "next": next_url, "logout": logout})
                elif not session_exists:
                    # Stale session, remove record
                    logged_in.delete()
                elif force_login:
                    # Delete other session and proceed
                    Session.objects.filter(session_key=logged_in.session_key).delete()
                    logged_in.delete()
            except LoggedInUser.DoesNotExist:
                pass

            login(request, user)
            request.session.save()
            LoggedInUser.objects.update_or_create(user=user, defaults={"session_key": request.session.session_key})
            
            safe_folder_name = user.email.replace("@", "_at_").replace(".", "_dot_")
            base_path = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name)
            for sub in ["profiles", "checkpoints", "saved_models"]:
                os.makedirs(os.path.join(base_path, sub), exist_ok=True)

            if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
                return redirect(next_url)
            return redirect("/")

        else:
            messages.error(request, "Invalid credentials")
            return render(request, "login.html", {"form": form, "next": next_url, "logout": logout})
        
    return render(request, "login.html", {"form": form, "next": next_url, "logout": logout})


def logout_all(request):
    try:
        print(request.user)
        LoggedInUser.objects.filter(username=request.user.username).delete()
    except:
        pass
    logout(request)
    return redirect('/login/')


def main(req):
    return render(req, "main_page.html", {"exp":None})