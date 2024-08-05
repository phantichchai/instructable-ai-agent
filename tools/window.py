import pygetwindow as gw

# Function to get the coordinates of a specific window
def get_window_coordinates(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    return {
        'left': window.left,
        'top': window.top,
        'width': window.width,
        'height': window.height
    }

