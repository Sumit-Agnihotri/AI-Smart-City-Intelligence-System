import os
import subprocess
import sys
from huggingface_hub import HfApi


def main():
    token = os.environ.get('HF_TOKEN')
    space = os.environ.get('HF_SPACE')  # expected like 'username/space-name'
    if not token or not space:
        print('HF_TOKEN and HF_SPACE env vars are required')
        sys.exit(2)

    api = HfApi()
    try:
        api.create_repo(repo_id=space, repo_type='space', token=token, private=False)
        print(f'Created space {space}')
    except Exception as e:
        print(f'Could not create space (might already exist): {e}')

    remote_url = f'https://{token}@huggingface.co/spaces/{space}.git'

    # Add remote and push
    try:
        subprocess.check_call(['git', 'remote', 'remove', 'hf'])
    except Exception:
        pass

    subprocess.check_call(['git', 'remote', 'add', 'hf', remote_url])
    # Push main branch
    subprocess.check_call(['git', 'push', '--force', 'hf', 'main'])


if __name__ == '__main__':
    main()
