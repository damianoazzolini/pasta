x(1..9).
y(1..9).
n(1..9).

{sudoku(X,Y,N): n(N)}=1 :- x(X) ,y(Y).

subgrid(X,Y,A,B) :- x(X), x(A), y(Y), y(B),(X-1)/3 == (A-1)/3, (Y-1)/3 == (B-1)/3.

:- sudoku(X,Y,N), sudoku(A,Y,N), X!=A.
:- sudoku(X,Y,N), sudoku(X,B,N), Y!=B.
:- sudoku(X,Y,V), sudoku(A,B,V), subgrid(X,Y,A,B), X != A, Y != B.

% #show sudoku/3.

abducible sudoku(1, 3, 6).
abducible sudoku(1, 8, 9).
abducible sudoku(2, 4, 5).
abducible sudoku(2, 6, 1).
abducible sudoku(2, 7, 7).
abducible sudoku(3, 1, 2).
abducible sudoku(3, 4, 9).
abducible sudoku(3, 7, 3).
abducible sudoku(4, 2, 7).

%
abducible sudoku(4, 5, 3).
%
abducible sudoku(4, 8, 5).
%
abducible sudoku(5, 2, 2).
%
abducible sudoku(5, 5, 9). 
%
abducible sudoku(5, 8, 6).
%
abducible sudoku(6, 2, 4).
%
abducible sudoku(6, 5, 8).
abducible sudoku(6, 8, 2).
abducible sudoku(7, 3, 1).
abducible sudoku(7, 6, 3).
abducible sudoku(7, 9, 4).
abducible sudoku(8, 4, 2).

abducible sudoku(8, 3, 5).
abducible sudoku(8, 6, 7).
abducible sudoku(9, 2, 3).

abducible sudoku(9, 7, 8).


% 0.5::sudoku(9, 7, 9).

qr:- sudoku(1,3,6).
