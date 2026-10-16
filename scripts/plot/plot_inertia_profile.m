function plot_inertia_profile(mat_path)
%PLOT_INERTIA_PROFILE  Plot 2-link robot-arm inertia profile from .mat file.
%
%   plot_inertia_profile(MAT_PATH)
%
%   Reads the .mat file produced by plot_inertia_profile.py and generates
%   a 3x2 subplot grid of all 6 unique inertia tensor components.
%
%   Layout:
%       Row 1: Jxx | Jxy
%       Row 2: Jyy | Jyz
%       Row 3: Jzz | Jzx
%
%   Example:
%       plot_inertia_profile('outputs/inertia_profile.mat')

    arguments
        mat_path (1,1) string
    end

    if ~isfile(mat_path)
        error('File not found: %s', mat_path);
    end

    data = load(mat_path);
    t       = data.time(:);
    J_upper = data.J_upper;          % (N, 6): [Jxx, Jyy, Jzz, Jxy, Jyz, Jzx]
    t_start = double(data.time_start_sec);
    t_dur   = double(data.duration_sec);
    t_end   = t_start + t_dur;

    % --- Labels & column mapping ---
    labels  = {'Jxx [kg m^2]', 'Jxy [kg m^2]'; ...
               'Jyy [kg m^2]', 'Jyz [kg m^2]'; ...
               'Jzz [kg m^2]', 'Jzx [kg m^2]'};
    col_map = [1 4; 2 5; 3 6];

    c_line = [0.12 0.37 0.65];       % steel blue

    fig = figure('Position', [100 100 600 450]);

    for row = 1:3
        for col = 1:2
            ax = subplot(3, 2, (row-1)*2 + col);
            hold(ax, 'on');

            idx = col_map(row, col);
            plot(ax, t, J_upper(:, idx), '-', ...
                 'Color', c_line, 'LineWidth', 1.5);

            ylabel(ax, labels{row, col});
            xlim(ax, [0, 50]);
            grid(ax, 'on');
            set(ax, 'GridAlpha', 0.15, 'TickDir', 'out', 'Box', 'on');

            if row == 3
                xlabel(ax, 'Time [s]');
            else
                set(ax, 'XTickLabel', []);
            end

            hold(ax, 'off');
        end
    end

    % Link x-axes
    all_ax = findobj(fig, 'Type', 'axes');
    linkaxes(all_ax, 'x');

    % --- Console summary ---
    fprintf('\n=== 2-Link Robot Arm Inertia Profile ===\n');
    fprintf('  Link 1: m=%.1f kg, L=%.2f m, theta %.0f -> %.0f deg\n', ...
            double(data.mass_1), double(data.arm_length_1), ...
            double(data.theta_1_0_deg), double(data.theta_1_f_deg));
    fprintf('  Link 2: m=%.1f kg, L=%.2f m, theta %.0f -> %.0f deg, delay=%.1f s\n', ...
            double(data.mass_2), double(data.arm_length_2), ...
            double(data.theta_2_0_deg), double(data.theta_2_f_deg), ...
            double(data.delay_sec));
    fprintf('  Transition: %.1f -- %.1f s\n', t_start, t_end);
    fprintf('=========================================\n');
end
