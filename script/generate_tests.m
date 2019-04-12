% Generates test cases based on the original Athey and Imbens (2006) code
% You need to place the content to calculate their tables in this directory
% for this script to run. (They are not added to this repository for
% licensing reasons). At the time of writing, these could be obtained from
% Susan Athey's website: https://athey.people.stanford.edu/research under
% the link "Download CIC code".

out_dir = "../tests/cases/";

% Set seed
rng(97854459);

for i = 1:10
    % Calculate quantile treatment effects at these points
    qq = [.1; .2; .3; .4; .5; .6; .7; .8; .9];

    % Sample size of every group is random int between 100 and 200
    n00 = floor(100 * rand()) + 100;
    n01 = floor(100 * rand()) + 100;
    n10 = floor(100 * rand()) + 100;
    n11 = floor(100 * rand()) + 100;

    [y00, y01, y10, y11] = gen(randn(), randn(), randn(), randn(), n00, n01, n10, n11);

    [est, se] = cic(y00, y01, y10, y11, qq, 1, 499);
    out_path = out_dir + sprintf("exp%d.mat", i);
    save(out_path, "y00", "y01", "y10", "y11", "est", "se");
end