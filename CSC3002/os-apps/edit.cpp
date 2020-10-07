/* Modified upon antirez's kilo editor
 * https://github.com/antirez/kilo
 * Copyright (C) 2016 Salvatore Sanfilippo <antirez at gmail dot com>
 */

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cerrno>
#include <cstring>
//#include "edit.h"

/*** defines ***/

#define TAB_SPACES 8
#define MAX_NUM_ROWS 10000
#define WELCOME "Text Editor (Press Ctrl-Q to quit)"

// VT100 escape sequences
#define CS "\x1b[2J" // Clear Screen
#define CL "\x1b[K" // Clear line (to the right of the cursor)
#define CUR_TL "\x1b[H" // Cursor TopLeft
#define CUR_HIDE "\x1b[?25l" // Cursor Hide
#define CUR_SHOW "\x1b[?25h" // Cursor Show
#define TXT_INV_COLOR "\x1b[7m" // Text Invert Color
#define TXT_CLR_ATTR "\x1b[m" // Text Cear Attributes

enum editorKey {
    BACKSPACE = 127,
    UP = 315,
    DOWN = 316,
    LEFT = 317,
    RIGHT = 318,
    PAGE_UP = 319,
    PAGE_DOWN = 320,
    HOME_KEY = 321,
    END_KEY = 322,
    DEL_KEY = 323,
};

/*** data ***/

// single row of text
struct editorRow {
    char* buff;
    //char* rbuff; // buff for actual char to render
    int size;
    //int rsize;
};

struct editorConfig {
    int cur_col, cur_row;           // cursor position within text
    //int rcur_col;                 // cursor column within rendered text
    int row_off, col_off;           // offsets
    int screen_rows, screen_cols;   // number of rows and cols on screen
    int num_rows;                   // number of editor rows
    struct editorRow* row;          // array of rows
    char* file_name;
    struct termios prev_termios;
    int total_len;
};

struct editorConfig E;

/*** append buffer ***/

struct abuff {
    char* buff;
    int len;
};

#define ABUFF_INIT {NULL, 0}

void buffAppend(struct abuff & ab, const char* s, int len) {
    char* new_buff = (char*) realloc(ab.buff, ab.len + len);

    if (new_buff == NULL) return;

    // appends string s to the end of the new buffer
    memcpy(&new_buff[ab.len], s, len);

    // updates buffer and length
    ab.buff = new_buff;
    ab.len += len;
}

void buffFree(struct abuff & ab) {
    free(ab.buff);
}

/*** terminal ***/

void resetScreen() {
    struct abuff ab = ABUFF_INIT;
    buffAppend(ab, CS, 4);
    buffAppend(ab, CUR_TL, 3);
    write(STDOUT_FILENO, ab.buff, ab.len);
    buffFree(ab);
}

int getWindowSize(int & s_rows, int & s_cols) {
    struct winsize ws;

    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1 || ws.ws_col == 0)
        return -1;
    s_cols = ws.ws_col;
    s_rows = ws.ws_row;
    return 0;
}

void error(const char* s) {
    resetScreen();
    perror(s);
    exit(1);
}

void resumePreviousMode() {
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &E.prev_termios) == -1)
        error("tcsetattr");
}

void enableRawMode() {
    // turns off raw mode at exit
    atexit(resumePreviousMode);

    // gets terminal's attributes
    if (tcgetattr(STDIN_FILENO, &E.prev_termios) == -1)
        error("tcgetattr");

    struct termios raw = E.prev_termios;

    // turns off  "\n"->"\r\n" translation
    raw.c_iflag &= ~(OPOST);
    //                ^M     ^S^Q
    raw.c_iflag &= ~(ICRNL | IXON);
    //               echo   cooked mode  ^C^Z     ^V
    raw.c_lflag &= ~(ECHO |   ICANON   | ISIG | IEXTEN);
    //               other fancy flags
    raw.c_lflag &= ~(BRKINT | INPCK | ISTRIP);
    // sets character size to 8 bits per byte
    raw.c_lflag |= (CS8);
    // read() returns the byte entered or zero if timeout
    raw.c_cc[VMIN] = 0;
    // max waiting time for read() = 100 ms
    raw.c_cc[VTIME] = 1;

    // sets terminal to raw mode
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == -1)
        error("tcsetattr");
}

int readKey() {
    int nread = 0;
    char c;
    while (nread != 1) {  // will re-read when timed out
        nread = read(STDIN_FILENO, &c, 1);
        if (nread == -1 && errno != EAGAIN)
            error("read");
    }

    if (c == '\x1b') {
        char seq[3];

        // maps escape sequence
        if (read(STDIN_FILENO, &seq, 2) == 2) {

            if (seq[0] == '[') {
                // arrow keys
                switch (seq[1]) {
                case 'A': return UP;
                case 'B': return DOWN;
                case 'C': return RIGHT;
                case 'D': return LEFT;
                case 'F': return END_KEY;
                case 'H': return HOME_KEY;
                }

                // Home, End, PgUp, PgDn
                if (seq[1] >= '0' && seq[1] <= '9' &&
                    read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == '~') {
                        switch(seq[1]) {
                        case '1':
                        case '7': return HOME_KEY;
                        case '3': return DEL_KEY;
                        case '4':
                        case '8': return END_KEY;
                        case '5': return PAGE_UP;
                        case '6': return PAGE_DOWN;
                        }
                }
            }

            if (seq[0] == 'O') {
                switch (seq[1]) {
                case 'F': return END_KEY;
                case 'H': return HOME_KEY;
                }
            }
        }
    }
    return c;
}

/*** row operations ***/

char* renderLine(char* line, ssize_t len, ssize_t & rlen) {
    // strips off '\n' and '\r' at the end of line
    while (len && (line[len - 1] == '\n' ||
                   line[len - 1] == '\r')) {
        --len;
    }

    // converts tabs to spaces
    int i, ri;
    rlen = len;
    for (i = 0; i < len; i++) {
        if (line[i] == '\t') rlen += (TAB_SPACES - 1);
    }

    char* rline = (char*) malloc(rlen);
    for (i = ri = 0; i < len; i++) {
        if (line[i] == '\t') {
            rline[ri++] = ' ';
            while ((ri - i) % TAB_SPACES)
                rline[ri++] = ' ';
        } else {
            rline[ri++] = line[i];
        }
    }

    return rline;
}

void insertRow(int at, char* s, size_t len) {
    if (at < 0 || at > E.num_rows) return;

    E.row = (struct editorRow*) realloc(E.row, sizeof (struct editorRow) * (E.num_rows + 1));
    for (int i = E.num_rows - 1; i >= at; i--) {
        memcpy(&E.row[i + 1], &E.row[i], sizeof (struct editorRow));
    }

    ssize_t rlen;
    char* rs = renderLine(s, len, rlen);
    E.row[at].size = rlen;
    E.row[at].buff = (char*) malloc(rlen + 1);
    mempcpy(E.row[at].buff, rs, rlen);
    E.row[at].buff[rlen] = '\0';
    E.total_len += rlen;

    ++E.num_rows;
}

void appendRow(char* s, size_t len) {insertRow(E.num_rows, s, len);}

void deleteRow(int at) {
    if (!(at < 0 || at >= E.num_rows)) {
        free(E.row[at].buff);
        for (int i = at + 1; i < E.num_rows; i++)
            memcpy(&E.row[i - 1], &E.row[i], sizeof (struct editorRow));
        --E.num_rows;
    }
}

void rowInsertChar(struct editorRow & row, int at, int c) {
    if (at < 0 || at > row.size) at = row.size;
    row.buff = (char*) realloc(row.buff, row.size + 2);
    for (int i = row.size; i >= at; i--)
        row.buff[i + 1] = row.buff[i];
    row.buff[at] = c;
    ++row.size;

    // index could be messed up...
    ssize_t rlen;
    char* rline = renderLine(row.buff, row.size + 1, rlen);
    row.buff = (char*) malloc(rlen);
    memcpy(row.buff, rline, rlen);
    row.size = rlen - 1;
}

void insertNewline() {
    if (E.cur_col == 0) {
        insertRow(E.cur_row, (char*)"", 0);
    } else {
        insertRow(E.cur_row + 1, &E.row[E.cur_row].buff[E.cur_col],
                  E.row[E.cur_row].size - E.cur_col);
        E.row[E.cur_row].size = E.cur_col;
        E.row[E.cur_row].buff[E.row[E.cur_row].size] = '\0';
        E.total_len -= E.row[E.cur_row + 1].size;
    }
    E.cur_col = 0;
    ++E.cur_row;
}

void rowAppendString(struct editorRow & row, char* s, size_t len) {
    ssize_t rlen;
    char* rs = renderLine(s, len, rlen);
    row.buff = (char*) realloc(row.buff, row.size + rlen + 1);
    memcpy(&row.buff[row.size], rs, rlen);
    row.size += rlen;
    row.buff[row.size] = '\0';
}

void rowDeleteChar(struct editorRow & row, int at) {
    if (!(at < 0 || at > row.size)) {
        for (int i = at; i < row.size; i++)
            row.buff[i] = row.buff[i + 1];
        --row.size;
    }
}

char* rowsToString(int & buff_len) {
    buff_len = E.total_len + E.num_rows;
    char* buff = (char*) malloc(buff_len);
    char* p = buff;
    for (int i = 0; i < E.num_rows; i++) {
        memcpy(p, E.row[i].buff, E.row[i].size);
        p += E.row[i].size;
        *p = '\n';
        ++p;
    }
    return buff;
}

/*** text editing ***/

void insertChar(int c) {
    if (E.cur_row == E.num_rows) {
        appendRow((char*)"", 0);
    }
    rowInsertChar(E.row[E.cur_row], E.cur_col, c);

    int rlen;
    switch (c) {
    case '\t':
        rlen = TAB_SPACES;
        break;
    default:
        rlen = 1;
    }

    E.cur_col += rlen;
    E.total_len += rlen;
}

void deleteChar() {
    if ((E.cur_row == E.num_rows) ||
        (E.cur_col == 0 && E.cur_row == 0)) return;

        if (E.cur_col > 0) {
            rowDeleteChar(E.row[E.cur_row], E.cur_col - 1);
            --E.cur_col;
            --E.total_len;
        } else {
            E.cur_col = E.row[E.cur_row - 1].size;
            rowAppendString(E.row[E.cur_row - 1], E.row[E.cur_row].buff, E.row[E.cur_row].size);
            deleteRow(E.cur_row);
            --E.cur_row;
        }
}

/*** file i/o ***/

void openFile(char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) error("fopen");
    E.file_name = (char*) malloc(strlen(filename));
    memcpy(E.file_name, filename, strlen(filename));

    char *line, *rline;
    size_t line_capacity = 0;
    ssize_t line_len, rline_len;
    while ( (line_len = getline(&line, &line_capacity, f) ) != -1) {
        rline = renderLine(line, line_len, rline_len);
        appendRow(rline, rline_len);
    }
    free(line);
    free(rline);
    fclose(f);
}

void saveFile() {
    if (E.file_name == NULL) return;

    int buff_len;
    char* buff = rowsToString(buff_len);
    int fd = open(E.file_name, O_RDWR | O_CREAT, 0644);
    if (fd != -1) {
        if (ftruncate(fd, buff_len) != -1) {
            if (write(fd, buff, buff_len) == buff_len) {
                close(fd);
                free(buff);
                return;
            }
        }
        close(fd);
    }
    free(buff);
}

/*** output ***/

// updates offsets based on cursor info
void scroll() {

    // cursor above screen
    if (E.cur_row < E.row_off) {
        E.row_off = E.cur_row;
        //--E.row_off;
    }

    // cursor below screen
    else if (E.cur_row >= E.row_off + E.screen_rows) {
        E.row_off = E.cur_row - E.screen_rows + 1;
        //++E.row_off;
    }

    // cursor left of screen
    if (E.cur_col < E.col_off) {
        E.col_off = E.cur_col;
    }

    // cursor right of screen
    else if (E.cur_col >= E.col_off + E.screen_cols) {
        E.col_off = E.cur_col - E.screen_cols + 1;
    }

}

void drawWelcome(struct abuff & ab) {
    int welcome_len = strlen(WELCOME) > E.screen_cols? E.screen_cols : strlen(WELCOME);
    int padding = (E.screen_cols - welcome_len) / 2;
    if (padding) {
        buffAppend(ab, "~", 1);
        --padding;
    }
    while (padding--) buffAppend(ab, " ", 1);
    buffAppend(ab, WELCOME, welcome_len);
}

void drawRows(struct abuff & ab) {
    int y, r;  // y: screen_coord, r: file_row
    for (y = 0; y < E.screen_rows; y++) {

        // draws texts in row[]
        r = y + E.row_off;
        if (r < E.num_rows) {
            int len = E.row[r].size - E.col_off;
            if (len < 0) len = 0;
            if (len > E.screen_cols) len = E.screen_cols;
            buffAppend(ab, &E.row[r].buff[E.col_off], len);
        }

        else if (E.num_rows == 0 && r == E.screen_rows / 3) {
            drawWelcome(ab);
        } else {
            buffAppend(ab, "~", 1);
        }

        // drawing done. clears what's left on the line and start a new one
        buffAppend(ab, CL, 3);
        buffAppend(ab, "\r\n", 2);
    }
}

void drawStatus(struct abuff & ab) {
  buffAppend(ab, TXT_INV_COLOR, 4);
  char status[100], rstatus[100];
  int status_len = snprintf(status, sizeof(status),
                     "%.20s - %d lines", E.file_name ? E.file_name : "[No Name]", E.num_rows);
  int rstatus_len = snprintf(rstatus, sizeof(rstatus),
                             "Line %d, Column %d (%d characters)", E.cur_row + 1, E.cur_col + 1, E.total_len);
  if (status_len > E.screen_cols) status_len = E.screen_cols;
  buffAppend(ab, status, status_len);
  while (status_len < E.screen_cols) {
    if (status_len == E.screen_cols - rstatus_len) {
        buffAppend(ab, rstatus, rstatus_len);
        break;
    }
    buffAppend(ab, " ", 1);
    ++status_len;
  }
  buffAppend(ab, TXT_CLR_ATTR, 3);
}

void refreshScreen() {
    // updates offset accordingly to updated cursor position
    scroll();

    struct abuff ab = ABUFF_INIT;

    buffAppend(ab, CUR_HIDE, 6);

    // draws contents from topleft
    buffAppend(ab, CUR_TL, 3);
    drawRows(ab);
    drawStatus(ab);

    // updates cursor position on screen
    char cur_esc[32];
    snprintf(cur_esc, sizeof(cur_esc), "\x1b[%d;%dH", E.cur_row - E.row_off + 1, E.cur_col - E.col_off + 1);
    buffAppend(ab, cur_esc, strlen(cur_esc));

    buffAppend(ab, CUR_SHOW, 6);

    // renders screen
    write(STDOUT_FILENO, ab.buff, ab.len);
    buffFree(ab);
}

/*** input ***/

void moveCursor(int c) {
    editorRow* row = E.cur_row < E.num_rows? &E.row[E.cur_row] : NULL;

    switch (c) {
    case UP:
        if (E.cur_row) --E.cur_row;
        break;
    case DOWN:
        if (E.cur_row < E.num_rows)
            ++E.cur_row;
        break;
    case LEFT:
        if (E.cur_col) --E.cur_col;
        else if (E.cur_row) {
            --E.cur_row;
            E.cur_col = E.row[E.cur_row].size;
        }
        break;
    case RIGHT:
        // if left of the end of the line, move cursor right
        if (row && E.cur_col < row->size)
            ++E.cur_col;
        else if (row && E.cur_col == row->size) {
            ++E.cur_row;
            E.cur_col = 0;
        }
        break;
    case END_KEY:
        if (E.cur_row < E.num_rows)
            E.cur_col = E.row[E.cur_row].size;
        break;
    case HOME_KEY:
        E.cur_col = 0;
        break;
    }

    if (E.cur_row < E.num_rows) {
        row = &E.row[E.cur_row];
        // snaps cursor to the right of the line
        if (E.cur_col > row->size) E.cur_col = row->size;
    } else E.cur_col = 0;
}

void processKeyPress() {
    int key = readKey();

    switch (key) {
    case UP:
    case DOWN:
    case LEFT:
    case RIGHT:
    case END_KEY:
    case HOME_KEY:
        moveCursor(key);
        break;

    case PAGE_UP:
    case PAGE_DOWN: {
        // places cursor correctly
        if (key == PAGE_UP) {
          E.cur_row = E.row_off;
        } else {
          E.cur_row = E.row_off + E.screen_rows - 1;
          if (E.cur_row > E.num_rows) E.cur_row = E.num_rows;
        }
        // scrolls an entire page
        int times = E.screen_rows;
        int dir = (key == PAGE_UP? UP : DOWN);
        while (times--)
            moveCursor(dir);
        break;
    }

    case BACKSPACE:
    case CTRL('h'):
    case DEL_KEY:
        if (key == DEL_KEY) moveCursor(RIGHT);
        deleteChar();
        break;

    case CTRL('s'):
        saveFile();
        break;

    case CTRL('q'):
        resetScreen();
        exit(0);
        break;

    case CTRL('l'):
    case '\x1b':
        break;

    case '\r':
        insertNewline();
        break;

    default:
        insertChar(key);
    }
}

/*** init ***/

void initEditorConfig() {
    if (getWindowSize(E.screen_rows, E.screen_cols) == -1)
        error("getWindowSize");
    --E.screen_rows;

    E.cur_row = E.cur_col = 0;
    E.row_off = E.col_off = 0;
    E.num_rows = 0;
    E.total_len = 0;
    E.row = (struct editorRow*) malloc(MAX_NUM_ROWS * sizeof(editorRow));
    E.file_name = NULL;
}

/*** main ***/

int main(int argc, char* argv[]) {
    enableRawMode();    // back to cooked mode at exit
    initEditorConfig();
    if (argc >= 2)      // if filename passed in
        openFile(argv[1]);
    refreshScreen();

    // after each keypress...
    while (true) {
        processKeyPress(); // updates cursor position; exit...

        if (getWindowSize(E.screen_rows, E.screen_cols) == -1)
            error("getWindowSize");
        --E.screen_rows;

        refreshScreen();
    }

    return 0;
}
