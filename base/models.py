from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class history(models.Model):
    user=models.ForeignKey(User, on_delete=models.CASCADE)
    translation_text=models.TextField()
    translation_date=models.DateTimeField(auto_now_add=True)


    def __str__(self) -> str:
        return f'{self.user.username} {self.translation_date}'
