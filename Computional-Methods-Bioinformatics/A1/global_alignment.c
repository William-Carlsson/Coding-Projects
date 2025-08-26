/*
 * To compile this C program, placing the executable file in 'global', type:
 *
 *      gcc -o global global_alignment.c
 *
 * To run the program, type:
 *
 *      ./global
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


int main()
{ 
	int	i, j;
	int	m, n;
	int	alignmentLength, score, tmp;
	char X[MAX_LENGTH+1] = "ATCGAT";
	char Y[MAX_LENGTH+1] = "ATACGT";

	int	F[MAX_LENGTH+1][MAX_LENGTH+1];		/* score matrix */
	int	trace[MAX_LENGTH+1][MAX_LENGTH+1];	/* trace matrix */
	char	alignX[MAX_LENGTH*2];		/* aligned X sequence */
	char	alignY[MAX_LENGTH*2];		/* aligned Y sequence */

	int matchCount = 0;    /* Counter for matching positions */
	double percentIdentity; /* Variable to store percent identity */
	int hammingDistance = 0;     /* Counter for differing positions (Hamming distance) */


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
	 * Initialise matrices
	 */

	F[0][0] = 0;
	trace[0][0] = STOP;
	for ( i=1 ; i<=m ; i++ ) {
		F[i][0] = F[i-1][0] - GAP_PENALTY;
		trace[i][0] = STOP;
	}
	for ( j=1 ; j<=n ; j++ ) {
		F[0][j] = F[0][j-1] - GAP_PENALTY;
		trace[0][j] = STOP;
	}


 	/*
	 * Fill matrices
	 */

	for ( i=1 ; i<=m ; i++ ) {

		for ( j=1 ; j<=n ; j++ ) {
	
			if ( X[i-1]==Y[j-1] ) {
				score = F[i-1][j-1] + MATCH_SCORE;
			} else {
				score = F[i-1][j-1] + MISMATCH_SCORE;
			}
			trace[i][j] = DIAG;

			tmp = F[i-1][j] - GAP_PENALTY;
			if ( tmp>score ) {
				score = tmp;
				trace[i][j] = UP;
			}

			tmp = F[i][j-1] - GAP_PENALTY;
			if( tmp>score ) {
				score = tmp;
				trace[i][j] = LEFT;
			}

			F[i][j] = score;
 		}
	}


	/*
	 * Print score matrix
	 */

	printf("Score matrix:\n      ");
	for ( j=0 ; j<n ; ++j ) {
		printf("%5c", Y[j]);
	}
	printf("\n");
	for ( i=0 ; i<=m ; i++ ) {
		if ( i==0 ) {
			printf(" ");
		} else {
			printf("%c", X[i-1]);
		}
		for ( j=0 ; j<=n ; j++ ) {
			printf("%5d", F[i][j]);
		}
		printf("\n");
	}
	printf("\n");


	/*
	 * Trace back from the lower-right corner of the matrix
	 */

	i = m;
	j = n;
	alignmentLength = 0;



	while ( trace[i][j] != STOP ) {

		switch ( trace[i][j] ) {

			case DIAG:
				alignX[alignmentLength] = X[i-1];
				alignY[alignmentLength] = Y[j-1];
				i--;
				j--;
				alignmentLength++;
				break;

			case LEFT:
				alignX[alignmentLength] = '-';
				alignY[alignmentLength] = Y[j-1];
				j--;
				alignmentLength++;
				break;

			case UP:
				alignX[alignmentLength] = X[i-1];
				alignY[alignmentLength] = '-';
				i--;
				alignmentLength++;
		}
	}


	/*
	 * Unaligned beginning
	 */

	while ( i>0 ) {
		alignX[alignmentLength] = X[i-1];
		alignY[alignmentLength] = '-';
		i--;
		alignmentLength++;
	}

	while ( j>0 ) {
		alignX[alignmentLength] = '-';
		alignY[alignmentLength] = Y[j-1];
		j--;
		alignmentLength++;
	}


	/*
	 * Print alignment
	 */

	for ( i=alignmentLength-1 ; i>=0 ; i-- ) {
		printf("%c",alignX[i]);
	}
	printf("\n");
	for ( i=alignmentLength-1 ; i>=0 ; i-- ) {
		if(alignX[i] == alignY[i]){
			printf("|");
			matchCount++; // Increment match count for identical pair
		} else{
			printf(" ");
			hammingDistance++; /* Increment hamming distance for each mismatch */
		}
	}

	printf("\n");
	for ( i=alignmentLength-1 ; i>=0 ; i-- ) {
		printf("%c",alignY[i]);
	}
	printf("\n");


	/*
	 * Calculate and print percent identity
	 * From the lecture: (Number of aligment positions with matchning characters / the length of the alignmnet) * 100
	 */
	percentIdentity = ((double)matchCount / alignmentLength) * 100;
	printf("\nPercent Identity: %.2f%%\n", percentIdentity);
	printf("Hamming Distance: %d\n", hammingDistance);

	return(1);
}
