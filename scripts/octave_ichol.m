% Load sparse matrix from matrix.mat
data = load("matrix.mat");
fields = fieldnames(data);
A = data.(fields{1}); % Assumes the matrix is the first variable in the .mat file

% Ensure A is sparse
A = sparse(A);

opts.type = "nofill"; % No fill-in
L_ichol = ichol(A, opts);
preconditioner = inv(L_ichol * L_ichol');
save("-mat7-binary", "L_ichol.mat", "preconditioner");
