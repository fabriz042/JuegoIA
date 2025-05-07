import curses
import time

# Inicializa la pantalla
stdscr = curses.initscr()
curses.curs_set(0)  # Oculta el cursor
height, width = stdscr.getmaxyx()  # Obtén el tamaño de la pantalla
window = curses.newwin(height, width, 0, 0)
window.keypad(1)
window.timeout(100)  # Tiempo de espera entre actualizaciones en milisegundos

# Posición inicial del cuadrado
x = width // 2
y = height - 2

# Bucle principal
while True:
    window.clear()  # Limpia la pantalla
    window.addstr(y, x, "Y")  # Dibuja el cuadrado en la posición actual
    window.refresh()  # Actualiza la pantalla
    
    y -= 1  # Mueve el cuadrado hacia arriba

    if y < 0:
        y = height - 2  # Resetea la posición cuando llega al tope

    time.sleep(0.1)  # Pausa para hacer el movimiento visible
