import os

commit_string = "modify code structure"
not_add = ['result', 'data', 'weights', 'results']
for item in os.listdir():
    if item in not_add:
        continue
    else:
        os.system(f"git add {item}")
os.system(f'git commit -m "{commit_string}"')
os.system("git push origin main")