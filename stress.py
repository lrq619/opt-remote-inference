import subprocess

def main(stressor_num):

    stressors = []
    for i in range(stressor_num):
        stressor_process = subprocess.Popen(['./stressor'])
        stressors.append(stressor_process)

    serving_process = subprocess.Popen(['python', 'generate.py'])


    serving_process.communicate()

    for stressor in stressors:
        stressor.kill()
    # stressor_process.communicate()

if __name__ == '__main__':
    # for i in range(1,9):
    main(1)