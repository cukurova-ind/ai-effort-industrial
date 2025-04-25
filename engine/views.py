from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session
from django.utils import timezone
from .models import Room


def main_board(req, profile_name=None):
    if req.user.is_authenticated:
        if profile_name:
            profile_name = profile_name

        return render(req, "trainboard.html", {"profile_name":profile_name})
    else:
        return HttpResponseRedirect("/login/?next=/engine/")

def room_view(request, room_name):
    chat_room, created = Room.objects.get_or_create(name=room_name)
    return render(request, 'room.html', {
        'room': chat_room,
    })
