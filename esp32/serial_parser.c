/*
 * serial_parser.ino
 *
 * ESP32-side parser for Raspberry Pi direction messages.
 *
 * Protocol:
 *   Raspberry Pi -> ESP32:
 *       D,n\n
 *       D,l\n
 *       D,m\n
 *       D,r\n
 *
 * Direction mapping:
 *   n -> 0  (no detection)
 *   l -> 1  (left)
 *   m -> 2  (middle)
 *   r -> 3  (right)
 *
 * This parser is designed to:
 * - read newline-terminated serial messages
 * - validate format safely
 * - update a compact integer state for robot control logic
 */

int detectionState = 0;      // 0 = none, 1 = left, 2 = middle, 3 = right
String serialBuffer = "";

const int MAX_BUFFER_LENGTH = 32;


/**
 * Convert a direction character into an integer control code.
 */
int directionToState(char direction) {
  switch (direction) {
    case 'n': return 0;
    case 'l': return 1;
    case 'm': return 2;
    case 'r': return 3;
    default:  return 0;
  }
}


/**
 * Process one complete protocol line.
 *
 * Expected format:
 *   D,<char>
 *
 * Example:
 *   D,l
 */
void handleRaspberryPiLine(const String& line) {
  if (line.length() < 3) return;
  if (line.charAt(0) != 'D') return;
  if (line.charAt(1) != ',') return;

  char direction = tolower(line.charAt(2));

  if (direction != 'n' && direction != 'l' && direction != 'm' && direction != 'r') {
    return;
  }

  detectionState = directionToState(direction);

  // Optional debug output:
  // Serial.print("[RX] ");
  // Serial.print(line);
  // Serial.print(" -> detectionState=");
  // Serial.println(detectionState);
}


/**
 * Read and parse incoming serial data from the Raspberry Pi.
 *
 * Call this frequently inside loop().
 */
void readRaspberryPiSerial() {
  while (Serial.available() > 0) {
    char incomingChar = (char)Serial.read();

    // Ignore carriage return
    if (incomingChar == '\r') {
      continue;
    }

    // End of line -> process buffered message
    if (incomingChar == '\n') {
      if (serialBuffer.length() > 0) {
        handleRaspberryPiLine(serialBuffer);
        serialBuffer = "";
      }
      continue;
    }

    // Append character with overflow protection
    if (serialBuffer.length() < MAX_BUFFER_LENGTH) {
      serialBuffer += incomingChar;
    } else {
      serialBuffer = "";
    }
  }
}