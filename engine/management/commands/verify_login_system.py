from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session
from engine.models import LoggedInUser
from engine.utils.user_folders import ensure_user_folders, user_folder_exists


class Command(BaseCommand):
    help = 'Test and verify login system and user folder creation'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test-user',
            type=str,
            help='Username to test folder creation for',
        )
        parser.add_argument(
            '--cleanup',
            action='store_true',
            help='Clean up stale sessions and login records',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=== Login System Verification ==='))
        
        if options['cleanup']:
            self.cleanup_stale_sessions()
            
        if options['test_user']:
            self.test_user_folders(options['test_user'])
            
        self.show_current_state()

    def cleanup_stale_sessions(self):
        """Clean up stale sessions and login records"""
        self.stdout.write("Cleaning up stale sessions...")
        
        # Get all session keys that still exist
        valid_sessions = set(Session.objects.values_list('session_key', flat=True))
        
        # Find LoggedInUser records with invalid sessions
        stale_records = LoggedInUser.objects.exclude(session_key__in=valid_sessions)
        stale_count = stale_records.count()
        
        if stale_count > 0:
            stale_records.delete()
            self.stdout.write(
                self.style.WARNING(f'Removed {stale_count} stale login records')
            )
        else:
            self.stdout.write("No stale records found")

    def test_user_folders(self, username):
        """Test folder creation for a specific user"""
        self.stdout.write(f"Testing folder creation for user: {username}")
        
        try:
            user = User.objects.get(username=username)
            
            # Check if folders exist
            exists = user_folder_exists(user)
            self.stdout.write(f"User folders exist: {exists}")
            
            # Ensure folders exist
            if ensure_user_folders(user):
                self.stdout.write(
                    self.style.SUCCESS(f'User folders verified/created for {username}')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'Failed to create folders for {username}')
                )
                
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User {username} does not exist')
            )

    def show_current_state(self):
        """Show current state of login system"""
        self.stdout.write("\n=== Current System State ===")
        
        # Show logged in users
        logged_in_users = LoggedInUser.objects.all()
        self.stdout.write(f"Total logged in users: {logged_in_users.count()}")
        
        for record in logged_in_users:
            session_exists = Session.objects.filter(session_key=record.session_key).exists()
            status = "VALID" if session_exists else "STALE"
            self.stdout.write(
                f"  {record.user.username}: {record.session_key} [{status}]"
            )
        
        # Show total sessions
        total_sessions = Session.objects.count()
        self.stdout.write(f"Total active sessions: {total_sessions}")