% Randomly select the number of states (5 or 10)
n = randi([5, 10]);

% Generate system matrices
A = randn(n, n); % Modify A to ensure stability
while any(real(eig(A)) >= 0)
    A = randn(n,n);
end
B = randn(n, 1); % Assuming single input
C = randn(1, n); % Assuming single output
D = zeros(1, 1); % Assuming D is zero

% Check for Stability
eigenvalues = eig(A);
isStable = all(real(eigenvalues) < 0);

% Step 3: Design a Controller for Closed-Loop Stability
% Desired pole locations for closed-loop
desired_poles1 = -abs(randn(1, n)) - 1; 
K = place(A, B, desired_poles1);

% Step 4: Closed-loop System Analysis
A = A - B*K;

% Check for Observability
O = obsv(A, C);
isObservable = rank(O) == n;

% Display results
disp(['System is Stable: ', num2str(isStable)]);
disp(['System is Observable: ', num2str(isObservable)]);

% Design the Luenberger Observer
% Choose poles for the observer
desired_poles = -1 * (1:n); % Assuming 'n' is the number of states
L = place(A', C', desired_poles)';

% Initialize variables
t = 0:0.01:20; % Time vector
u = ones(length(t), 1); % Column vector for input

% Define the time step for the simulation
dt = 0.01;

% Initialize residuals variable
residuals = zeros(length(t), 1);

x = zeros(n, 1); % Reset initial state
x_hat = zeros(n, 1); % Reset initial estimated state
    for k = 1:length(t)
        w = 0.01 * randn(n, 1); % Process noise vector
        v = 0.01 * randn; % Measurement noise
        % Update actual system
        x_dot = A * x + B * u(k) + w;
        x = x + x_dot * dt;
        y = C * x + D * u(k) + v;
        % Update observer
        y_hat = C * x_hat + D * u(k); % Observer's output estimate
        x_hat_dot = A * x_hat + B * u(k) + L * (y - y_hat);
        x_hat = x_hat + x_hat_dot * dt;
        % Calculate residual
        residuals(k) = (C*x) - (C*x_hat);
    end

% Calculate standard deviation of residuals
std_residual = std(residuals(:));

% Set the threshold using 3-sigma rule
threshold = 3 * std_residual;

% Display the threshold value
disp(['Threshold: ', num2str(threshold)]);
% sys = ss(A, B, C, D);
% lsim(sys, u, t)
% Average the residuals (optional)
average_residuals = mean(residuals, 1);

% Plotting
figure;
plot(t, average_residuals);
xlabel('Time (seconds)');
ylabel('Residual');
title('Residuals Over Time');

% Modify the system matrix A to simulate a fault
A_faulty = A;  % Copy the original A matrix
A_faulty(1,1) = A_faulty(1,1) * 2;  % Example of a modification

% Run simulation with modified A matrix
x = zeros(n, 1); % Reset initial state
x_hat = zeros(n, 1); % Reset initial estimated state
for k = 1:length(t)
    w = 0.01 * randn(n, 1); % Process noise vector
    v = 0.01 * randn; % Measurement noise
    % Update actual system with modified A matrix
    x_dot = A_faulty * x + B * u(k) + w;
    x = x + x_dot * dt;
    y = C * x + D * u(k) + v;
    % Update observer (note: observer still uses the original A matrix)
    y_hat = C * x_hat + D * u(k); 
    x_hat_dot = A * x_hat + B * u(k) + L * (y - y_hat);
    x_hat = x_hat + x_hat_dot * dt;
end
% Calculate residual
residuals_faulty = (C*x) - (C*x_hat);
disp(['Residual_faulty: ', num2str(residuals_faulty)]);
% Check if any residual exceeds the threshold and trigger an alarm
if any(abs(residuals_faulty) > threshold)
    disp('Alarm! Potential fault detected.');
else
    disp('No fault detected.');
end
