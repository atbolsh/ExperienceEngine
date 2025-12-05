import os

WORKING_DIR = 'working'
WRONG_NAME = os.path.join(WORKING_DIR, 'anotated_view.jpg')

if os.path.exists(WRONG_NAME):
    try:
        os.remove(WRONG_NAME)
        print('Removed', WRONG_NAME)
    except Exception as e:
        print('Failed to remove', WRONG_NAME, e)
else:
    print('No wrong-named file present')
