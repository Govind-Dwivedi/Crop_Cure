from django.db import models

class Leaf(models.Model):
	leaf_img = models.ImageField(upload_to='images/')
