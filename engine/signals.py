from django.contrib.auth import user_logged_in, user_logged_out
from django.dispatch import receiver
from .models import LoggedInUser

# Disabled signals to avoid conflict with manual login view implementation
# The login view handles LoggedInUser creation/deletion manually

# @receiver(user_logged_in)
# def on_user_logged_in(sender, request, **kwargs):
#     LoggedInUser.objects.get_or_create(user=kwargs.get('user')) 

# @receiver(user_logged_out)
# def on_user_logged_out(sender, **kwargs):
#     LoggedInUser.objects.filter(user=kwargs.get('user')).delete()