from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

APP_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get("SECRET_KEY")

DEBUG = bool(os.environ.get("DEBUG", default=0))

ALLOWED_HOSTS = ["*"]

CORS_ALLOW_ALL_ORIGINS = True

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'industrial',
    'dataops',
    'prompting',
    'modeling',
    'channels',
    'engine.apps.EngineConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    
    'django.contrib.messages.middleware.MessageMiddleware',
    
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    #----------------------------------
    'engine.middleware.OneSessionPerUserMiddleware',
    #----------------------------------
]

ROOT_URLCONF = 'industrial.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'industrial.wsgi.application'
ASGI_APPLICATION = 'industrial.asgi.application'

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

DATABASES = {
    'default': {
         'ENGINE': os.getenv('DATABASE_ENGINE'),
         'NAME': os.getenv('DATABASE_NAME'),
         'USER': os.getenv('DATABASE_USERNAME'),
         'PASSWORD': os.getenv('DATABASE_PASSWORD'),
         'HOST': os.getenv('DATABASE_HOST'),
         'PORT': os.getenv('DATABASE_PORT', 5432),
     }
}

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',  # fast in-memory, good for development
        'LOCATION': 'unique-dataset-cache',
        'TIMEOUT': None,  # Cache forever; set a value (e.g., 300) to expire in seconds
    }
}

SESSION_EXPIRE_AT_BROWSER_CLOSE = True

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

STATIC_URL = 'static/'

LOGIN_URL = "/login/"

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

STATIC_ROOT =  os.path.join(BASE_DIR, 'static')

ENG_URL =  os.path.join(APP_DIR, 'engineering-torch')

MEDIA_URL = '/media/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

DATA_UPLOAD_MAX_NUMBER_FILES = 20000
