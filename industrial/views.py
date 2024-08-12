import os
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings


def main(req):
    return render(req, "main_page.html", {"exp":None})