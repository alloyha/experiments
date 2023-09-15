# Generated by Django 4.2.4 on 2023-08-27 13:01

from django.db.migrations import CreateModel
from django.db.models import BigAutoField, CharField, TextField, DateTimeField

# Post model
post_fields=[
    (
        "id",
        BigAutoField(
            auto_created=True,
            primary_key=True,
            serialize=False,
            verbose_name="ID",
        ),
    ),
    ("title", CharField(max_length=200)),
    ("content", TextField()),
    ("pub_date", DateTimeField(verbose_name="date published")),
]
post_model_args={
    "name": "Post",
    "fields": post_fields,
}
post_operation=CreateModel(
            name=post_model_args["name"],
            fields=post_model_args["fields"],
        )

# Available migrations
class Migration(Migration):

    initial = True

    dependencies = []

    operations = [ post_operation, ]