
import time


def servo_to_right():

    return 1

def servo_to_left():

    return 0

def servo_to_center():

    return 2

def buzzer_H_on():

    return 1

def buzzer_H_off():

    return 0
    
def buzzer_L_on():
    
    return 1

def buzzer_L_off():

    return 0


def read():
    
    return 4


def break_loop():

    return 0


while True:


    GOAL = read()
    Penalty = read()


    if Penalty < 200:

        servo_to_center()
        buzzer_H_off()
        buzzer_L_off()


    elif Penalty > 200:


        time_start = time.time()
        time_start_tension = time.time()

        t_pen = 0
        t_tension = 0

        while t_pen < 5:

            Penalty = read()
            GOAL = read()

            if GOAL < 200:


                if t_tension < 0.5: #s 

                    servo_to_left()
                    buzzer_H_on()
                    buzzer_L_off()

                elif t_tension > 0.5 and t_tension < 1: # s 

                    servo_to_center()
                    buzzer_H_off()
                    buzzer_L_off()

                elif t_tension > 1 and t_tension < 1.5: # s

                    servo_to_right()
                    buzzer_H_off()
                    buzzer_L_off()

                elif t_tension > 1.5 and t_tension < 2: # s

                    servo_to_center()
                    buzzer_H_on()
                    buzzer_L_off()

                elif t_tension > 2 and t_tension < 2.5: # s

                    servo_to_left()
                    buzzer_H_off()
                    buzzer_L_off()

                elif t_tension > 2.5 and t_tension < 3: # s

                    servo_to_center()
                    buzzer_H_off()
                    buzzer_L_off()

                elif t_tension > 3 and t_tension < 3.5: # s

                    servo_to_right()
                    buzzer_H_off()
                    buzzer_L_on()

                elif t_tension > 3.5: # s

                    time_start_tension = time.time()
                    t_tension = 0

            elif GOAL > 200:

                time_start_goal = time.time()

                t_goal = 0

                while t_goal < 4:


                    Penalty = read()


                    if Penalty < 200:

                        if t_goal < 0.5: #s 

                            servo_to_left()
                            buzzer_H_on()
                            buzzer_L_off()

                        elif t_goal > 0.5 and t_goal < 1: # s 

                            servo_to_center()
                            buzzer_H_off()
                            buzzer_L_off()

                        elif t_goal > 1 and t_goal < 1.5: # s

                            servo_to_right()
                            buzzer_H_off()
                            buzzer_L_off()

                        elif t_goal > 1.5 and t_goal < 2: # s

                            servo_to_center()
                            buzzer_H_on()
                            buzzer_L_off()

                        elif t_goal > 2 and t_goal < 2.5: # s

                            servo_to_left()
                            buzzer_H_off()
                            buzzer_L_off()

                        elif t_goal > 2.5: # s

                            time_start_goal = time.time()
                            t_goal = 0

                    elif Penalty > 200:

                        break_loop()


                    t_goal = time.time() - time_start_goal


            t_tension = time.time() - time_start_tension
            t_pen = time.time() - time_start 

    




