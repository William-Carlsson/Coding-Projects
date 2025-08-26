/*
 * To compile this C program, placing the executable file in 'levenshtein', type:
 *
 *      gcc -o levenshtein levenshtein.c
 *
 * To run the program, type:
 *
 *      ./levenshtein
 */

#include <stdio.h>

#define MAX_LENGTH	100

#define MATCH_SCORE	2
#define MISMATCH_SCORE	-1
#define GAP_PENALTY	2

#define STOP		0
#define UP			1
#define LEFT		2
#define DIAG		3


int min(int a, int b, int c) {
    int min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}


int main()
{ 
	int	i, j;
	int	m, n;
	int	alignmentLength, score, tmp;
	char	X[MAX_LENGTH+1] = "ATCGAT";
	char	Y[MAX_LENGTH+1] = "ATACGT";

	int F[MAX_LENGTH+1][MAX_LENGTH+1];		/* Levenshtein distance matrix */

	/*
	 * Find lengths of (null-terminated) strings X and Y
	 */
	m = 0;
	n = 0;
	while ( X[m] != 0 ) {
		m++;
	}
	while ( Y[n] != 0 ) {
		n++;
	}

	/*
	 * Initialize the matrix with the edit costs for transforming into empty strings
	 */
	for ( i = 0 ; i <= m ; i++ ) {
		F[i][0] = i; // Deletion cost
	}
	for ( j = 0 ; j <= n ; j++ ) {
		F[0][j] = j; // Insertion cost
	}

 	/*
	 * Fill matrix using the Levenshtein distance formula
	 */
	for ( i = 1 ; i <= m ; i++ ) {
		for ( j = 1 ; j <= n ; j++ ) {
			int cost = (X[i-1] == Y[j-1]) ? 0 : 1; // No cost if characters match, else substitution cost of 1
			F[i][j] = min(
				F[i-1][j] + 1,    // Deletion
				F[i][j-1] + 1,    // Insertion
				F[i-1][j-1] + cost // Substitution
			);
		}
	}

	/*
	 * Print the distance matrix
	 */
	printf("Levenshtein Distance Matrix:\n      ");
	for ( j = 0 ; j < n ; ++j ) {
		printf("%5c", Y[j]);
	}
	printf("\n");
	for ( i = 0 ; i <= m ; i++ ) {
		if ( i == 0 ) {
			printf(" ");
		} else {
			printf("%c", X[i-1]);
		}
		for ( j = 0 ; j <= n ; j++ ) {
			printf("%5d", F[i][j]);
		}
		printf("\n");
	}
	printf("\n");

    printf(X);
    printf("\n");
    printf(Y);
    printf("\n");

	/*
	 * Print the Levenshtein distance
	 */
	printf("Levenshtein Distance: %d\n", F[m][n]);

	return 0;
}
