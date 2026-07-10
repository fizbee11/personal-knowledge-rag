recipe-name:
  echo 'This is a recipe!'

# this is a comment
another-recipe:
  @echo 'This is another recipe.'


# index
index:
  python3 index_docs.py

run:
  python3 main.py

# source python venv
source:
  fish source /opt/python-venv/bin/activate.fish
