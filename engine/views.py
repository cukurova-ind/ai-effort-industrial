from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session
from django.utils import timezone
from .models import Room


def index_view(req):
    print(get_current_users())
    return render(req, "index.html")

def room_view(request, room_name):
    chat_room, created = Room.objects.get_or_create(name=room_name)
    return render(request, 'room.html', {
        'room': chat_room,
    })

def get_current_users():
    active_sessions = Session.objects.all()
    user_id_list = []
    for session in active_sessions:
        data = session.get_decoded()
        print(data)
        user_id_list.append(data.get('_auth_user_id', None))
    # Query all logged in users based on id list
    return User.objects.filter(id__in=user_id_list)