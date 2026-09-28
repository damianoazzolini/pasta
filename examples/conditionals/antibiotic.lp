% Consider a medical domain where we want to know whether to treat a patient with a specific antibiotic.
% In this scenario, we may be uncertain whether the patients have an infection (infection/1) and whether they have an allergy to the antibiotic (allergy/1).
% Furthermore, we have statistical information about the domain:
% 1) at least 30\% of people who show a high white blood cell (WBC) count have an infection and
% 2) at least 70\% of people with a high WBC count, infection, and without allergy should be treated with the antibiotic.
% This scenario (assuming, for simplicity, only 2 people) can be described with the following PASTA program

0.9::infection(1).
0.2::allergy(1).
0.8::infection(2).
0.1::allergy(2).

(elevated_wbc(X) | infection(X))[0.3,1].

(antibiotic(X) | elevated_wbc(X), infection(X), not allergy(X))[0.7,1].

qr:- antibiotic(1).

% pastasolver a.lp --query="qr"
% Lower probability for the query: 0.144
% Upper probability for the query: 0.72