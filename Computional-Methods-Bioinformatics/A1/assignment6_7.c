/*
 * To compile this C program, placing the executable file in 'global', type:
 *
 *      gcc -o assignment6_7 assignment6_7.c
 *
 * To run the program, type:
 *
 *      ./assignment6_7
 */

#include <stdio.h>
#include <string.h>

#define MAX_LENGTH   100

#define MATCH_SCORE  2
#define MISMATCH_SCORE -1
#define GAP_PENALTY  2

#define STOP        0
#define UP          1
#define LEFT        2
#define DIAG        3

void printAlignment(char *alignX, char *alignY, int length) {
    for (int i = length - 1; i >= 0; i--) printf("%c", alignX[i]);
    printf("\n");
    for (int i = length - 1; i >= 0; i--) printf("%c", alignY[i]);
    printf("\n\n");
}

void findAlignments(int i, int j, char *alignX, char *alignY, int alignmentLength,
                    int F[MAX_LENGTH+1][MAX_LENGTH+1], char X[], char Y[],
                    int trace[MAX_LENGTH+1][MAX_LENGTH+1], int *alignmentCount) {

    if (i == 0 && j == 0) {  // Base case: reached the top-left corner
        (*alignmentCount)++;  // Increment alignment count
        printAlignment(alignX, alignY, alignmentLength);
        return;
    }

    if (trace[i][j] == DIAG || (F[i][j] == F[i-1][j-1] + (X[i-1] == Y[j-1] ? MATCH_SCORE : MISMATCH_SCORE))) {
        alignX[alignmentLength] = X[i-1];
        alignY[alignmentLength] = Y[j-1];
        findAlignments(i-1, j-1, alignX, alignY, alignmentLength+1, F, X, Y, trace, alignmentCount);
    }

    if (trace[i][j] == UP || (F[i][j] == F[i-1][j] - GAP_PENALTY)) {
        alignX[alignmentLength] = X[i-1];
        alignY[alignmentLength] = '-';
        findAlignments(i-1, j, alignX, alignY, alignmentLength+1, F, X, Y, trace, alignmentCount);
    }

    if (trace[i][j] == LEFT || (F[i][j] == F[i][j-1] - GAP_PENALTY)) {
        alignX[alignmentLength] = '-';
        alignY[alignmentLength] = Y[j-1];
        findAlignments(i, j-1, alignX, alignY, alignmentLength+1, F, X, Y, trace, alignmentCount);
    }
}


int main() {
    int i, j;
    int m, n;
    int score, tmp;
    char X[MAX_LENGTH+1] = "ATTA";
    char Y[MAX_LENGTH+1] = "ATTTTA";

    int F[MAX_LENGTH+1][MAX_LENGTH+1];      /* score matrix */
    int trace[MAX_LENGTH+1][MAX_LENGTH+1];   /* trace matrix */
    char alignX[MAX_LENGTH*2];               /* aligned X sequence */
    char alignY[MAX_LENGTH*2];               /* aligned Y sequence */

    int alignmentCount = 0;  // Variable to count total number of alignments

    /* Find lengths of (null-terminated) strings X and Y */
    m = strlen(X);
    n = strlen(Y);

    /* Initialise matrices */
    F[0][0] = 0;
    trace[0][0] = STOP;

    for (i = 1; i <= m; i++) {
        F[i][0] = F[i-1][0] - GAP_PENALTY;
        trace[i][0] = UP;
    }
    for (j = 1; j <= n; j++) {
        F[0][j] = F[0][j-1] - GAP_PENALTY;
        trace[0][j] = LEFT;
    }

    /* Fill matrices */
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            int diagScore = F[i-1][j-1] + ((X[i-1] == Y[j-1]) ? MATCH_SCORE : MISMATCH_SCORE);
            int upScore = F[i-1][j] - GAP_PENALTY;
            int leftScore = F[i][j-1] - GAP_PENALTY;

            score = diagScore;
            trace[i][j] = DIAG;

            if (upScore > score) {
                score = upScore;
                trace[i][j] = UP;
            }
            if (leftScore > score) {
                score = leftScore;
                trace[i][j] = LEFT;
            }

            F[i][j] = score;
        }
    }

    /* Print score matrix */
    printf("Score matrix:\n      ");
    for (j = 0; j < n; ++j) printf("%5c", Y[j]);
    printf("\n");
    for (i = 0; i <= m; i++) {
        if (i == 0) printf(" ");
        else printf("%c", X[i-1]);
        for (j = 0; j <= n; j++) {
            printf("%5d", F[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    /* Print all optimal alignments */
    printf("All optimal alignments:\n\n");
    findAlignments(m, n, alignX, alignY, 0, F, X, Y, trace, &alignmentCount);

    /* Print total count of optimal alignments */
    printf("Total number of optimal alignments: %d\n", alignmentCount);

    return 0;
}

