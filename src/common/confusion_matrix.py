from matplotlib import pyplot

def create_alphabet(test_predictions, test_correct):
    # Generate headers for x and y columns
    def index_to_char(index):
        return str(chr(ord('A') + index))

    headers = list(map(index_to_char,range(0,26)))
    headers.extend(["="])

    create(test_predictions, test_correct, headers)

def create_greek(test_predictions, test_correct):
    # Generate headers for x and y columns
    alphabet = { 0: "π", 1: "α", 2: "β", 3: "σ", 4: "γ", 5: "δ", 6: "λ", 7: "ω", 8: "μ", 9: "ξ"}

    def index_to_char(index):
        return alphabet.get(index)

    headers = list(map(index_to_char, range(0,10)))
    headers.extend(["="])

    create(test_predictions, test_correct, headers)

def create(test_predictions, test_correct, headers):
    # Count occurrences to fill out values of the matrix
    headers_length = len(headers)
    alphabet_length = headers_length - 1

    values = [[0 for x in range(headers_length)] for y in range(headers_length)]

    for i in range(0, len(test_predictions)):
        correct = test_correct[i]
        predicted = test_predictions[i]
        values[predicted][correct] += 1

    # Sum up totals for rows and columns
    total = 0
    for i in range(alphabet_length):
        sum_horizontal = 0
        sum_vertical = 0
        for j in range(alphabet_length):
            sum_horizontal += values[i][j]
            sum_vertical += values[j][i]
        values[i][alphabet_length] = sum_horizontal
        values[alphabet_length][i] = sum_vertical
        total += sum_horizontal
    values[alphabet_length][alphabet_length] = total

    # Cell colors
    cellColours = [['#ffffff' for x in range(headers_length)] for y in range(headers_length)] # default to white
    for i in range(headers_length):
        for j in range(headers_length):
            if i == headers_length - 1 or j == headers_length - 1:
                cellColours[i][j] = '#929591' # grey for totals
            elif i == j and values[i][j] > 0:
                cellColours[i][j] = '#0165fc' # blue for correct predictions
            elif i == j and values[i][j] == 0:
                cellColours[i][j] = '#ff474c' # red for incorrect predictions
            elif values[i][j] > 0:
                cellColours[i][j] = '#ff474c' # red for incorrect predictions
            

    # Create and show the table
    table = pyplot.table(cellText=values, rowLabels = headers, colLabels=headers, loc='center', cellColours = cellColours)
    pyplot.axis('off')
    pyplot.show()