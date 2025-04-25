import os
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.sessions.models import Session
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme
from engine.models import LoggedInUser
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm


def custom_login_view(request):
    next_url = request.GET.get('next') or request.POST.get('next')
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user:
            try:
                logged_in = LoggedInUser.objects.get(user=user)
                if Session.objects.filter(session_key=logged_in.session_key).exists():
                    messages.error(request, "You are already logged in from another device or browser.")
                    return render(request, "login.html", {"form": form, "next": next_url})
                else:
                    # Stale session, remove record
                    logged_in.delete()
            except LoggedInUser.DoesNotExist:
                pass

            login(request, user)
            
            safe_folder_name = user.email.replace("@", "_at_").replace(".", "_dot_")
            safe_profile_folder = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name, "profiles")
            if not os.path.exists(safe_profile_folder):
                os.makedirs(safe_profile_folder)

            if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
                return redirect(next_url)
            return redirect("/")

        else:
            messages.error(request, "Invalid credentials")
            return render(request, "login.html", {"form": form, "next": next_url})
        
    return render(request, "login.html", {"form": form, "next": next_url})
    
def main(req):
    return render(req, "main_page.html", {"exp":None})