function plot_compare_boresight(run_id, varargin)
%PLOT_COMPARE_BORESIGHT  Compare boresight trajectories from a comparison run.
%
%   plot_compare_boresight(RUN_ID)
%   plot_compare_boresight(RUN_ID, 'Episode', EP, 'SplitBoresight', true)
%
%   Inputs:
%       RUN_ID  - date/time string, e.g. '2026-03-11/14-22-33'
%
%   Name-Value Options:
%       'Episode'        - episode index to plot (default: 1)
%       'SplitBoresight' - if true, one figure per boresight (default: false)
%
%   Example:
%       plot_compare_boresight('2026-03-11/14-22-33');
%       plot_compare_boresight('2026-03-11/14-22-33', 'SplitBoresight', true);

    % --- Parse inputs ---
    p = inputParser;
    addRequired(p, 'run_id', @ischar);
    addParameter(p, 'Episode', 1, @isnumeric);
    addParameter(p, 'SplitBoresight', false, @islogical);
    parse(p, run_id, varargin{:});

    ep = p.Results.Episode;
    split_bs = p.Results.SplitBoresight;

    % --- Fixed controller order and display names ---
    controller_order = {'nmpc', 'lsmpc', 'ukf_smpc', 'asgp_fast', 'party'};
    display_names = containers.Map( ...
        {'nmpc', 'lsmpc', 'ukf_smpc', 'asgp_fast', 'party'}, ...
        {'NMPC', 'L-SMPC', 'UKF-SMPC', 'ASGP-SMPC', 'CAK-SMPC'});

    % --- Method colors & styles (same as plot_compare_timeseries) ---
    method_colors = [
        1  0  0;   % nmpc      — 'r' red
        1.0, 0.6, 0.7;       % lsmpc
        0  0.8  0;   % ukf_smpc
        0  0  0;   % asgp_smpc — 'k' black
        0  0  1;   % party     — 'b' blue (ours)
    ];
    line_styles = {'-', '--', '-.', ':', '-'};
    line_widths = [1.75, 1.75, 1.75, 1.75, 1.75];
    ds_step = 5;  % downsample step for 3D line style visibility

    % KOZ cone colors
    koz_colors = [
        0.85  0.15  0.15;   % red
        0.15  0.15  0.85;   % blue
        0.85  0.55  0.10;   % orange
        0.15  0.75  0.35;   % green
    ];

    % --- Resolve base path ---
    base_path = fullfile('outputs', 'comparison', run_id);
    if ~isfolder(base_path)
        error('Run directory not found: %s', base_path);
    end

    % --- Discover experiment folders ---
    d = dir(base_path);
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..', '.hydra'}));
    exp_names = sort({d.name});

    if isempty(exp_names)
        error('No experiment folders found in %s', base_path);
    end

    % --- Show selection menu ---
    fprintf('\nAvailable experiments in %s:\n', run_id);
    for i = 1:numel(exp_names)
        % Count available controllers
        n_ctrl = 0;
        for c = 1:numel(controller_order)
            mat_path = fullfile(base_path, exp_names{i}, controller_order{c}, ...
                                'data', 'boresight_data.mat');
            if isfile(mat_path)
                n_ctrl = n_ctrl + 1;
            end
        end
        fprintf('  %d. %s  (%d controllers)\n', i, exp_names{i}, n_ctrl);
    end
    fprintf('  %d. ALL\n', numel(exp_names) + 1);
    fprintf('\n');

    idx = input('Select experiment number: ');
    if idx == numel(exp_names) + 1
        selected_exps = exp_names;
    elseif idx >= 1 && idx <= numel(exp_names)
        selected_exps = exp_names(idx);
    else
        error('Invalid selection: %d', idx);
    end

    % --- Process each selected experiment ---
    for e = 1:numel(selected_exps)
        exp_name = selected_exps{e};
        exp_path = fullfile(base_path, exp_name);

        % Discover controllers in fixed order
        mat_filepaths = {};
        method_labels = {};
        color_indices = [];
        for c = 1:numel(controller_order)
            ctrl_key = controller_order{c};
            mat_path = fullfile(exp_path, ctrl_key, 'data', 'boresight_data.mat');
            if isfile(mat_path)
                mat_filepaths{end+1} = mat_path; %#ok<AGROW>
                method_labels{end+1} = display_names(ctrl_key); %#ok<AGROW>
                color_indices(end+1) = c; %#ok<AGROW>
            end
        end

        N_methods = numel(mat_filepaths);
        if N_methods < 1
            fprintf('Skipping %s: no controllers found.\n', exp_name);
            continue;
        end

        fprintf('\nPlotting %s (%d controller(s): %s)\n', ...
                exp_name, N_methods, strjoin(method_labels, ', '));

        % --- Load all data ---
        all_data = cell(1, N_methods);
        for m = 1:N_methods
            all_data{m} = load(mat_filepaths{m});
            n_ep = double(all_data{m}.num_episodes);
            assert(ep <= n_ep, ...
                'Episode %d requested but %s has only %d episodes.', ...
                ep, method_labels{m}, n_ep);
        end

        % Use first file for KOZ geometry and goal
        ref = all_data{1};
        bs_names   = cellstr(ref.boresight_names);
        koz_names  = cellstr(ref.koz_names);
        koz_dirs   = ref.koz_directions;
        koz_angles = ref.koz_angles_deg;
        q_goal     = ref.q_goal(:)';

        N_bs = numel(bs_names);

        if split_bs
            % === One figure per boresight ===
            for b = 1:N_bs
                fig = figure('Name', sprintf('[%s] %s', exp_name, bs_names{b}), ...
                             'Position', [100 + 30*b, 100 + 30*b, 650, 500]);
                ax = axes(fig); hold(ax, 'on');

                draw_sphere(ax);
                draw_koz_cones(ax, koz_dirs, koz_angles, koz_names, koz_colors);

                % Goal marker
                bs_vec_B = ref.boresight_vecs_B(b, :);
                bs_goal = quat_rotate(q_goal, bs_vec_B);
                plot3(ax, bs_goal(1), bs_goal(2), bs_goal(3), ...
                      'p', 'Color', [0.85 0.65 0.0], 'MarkerSize', 18, ...
                      'MarkerFaceColor', [1 0.84 0], 'LineWidth', 1.2, ...
                      'HandleVisibility', 'off');

                % Start marker (same for all controllers)
                d_ref = all_data{1};
                bs_traj_ref = d_ref.(bs_names{b});
                x0 = squeeze(bs_traj_ref(ep, :, 1));
                y0 = squeeze(bs_traj_ref(ep, :, 2));
                z0 = squeeze(bs_traj_ref(ep, :, 3));
                v0 = find(~isnan(x0), 1);
                plot3(ax, x0(v0), y0(v0), z0(v0), ...
                      'o', 'Color', 'k', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
                      'LineWidth', 1.5, 'HandleVisibility', 'off');

                % Method trajectories
                for m = 1:N_methods
                    d_m = all_data{m};
                    ci = color_indices(m);
                    mc = method_colors(ci, :);
                    ls = line_styles{ci};
                    lw = line_widths(ci);

                    bs_traj = d_m.(bs_names{b});
                    x = squeeze(bs_traj(ep, :, 1));
                    y = squeeze(bs_traj(ep, :, 2));
                    z = squeeze(bs_traj(ep, :, 3));
                    valid = ~isnan(x);
                    xv = x(valid); yv = y(valid); zv = z(valid);
                    idx_ds = [1:ds_step:numel(xv), numel(xv)];

                    plot3(ax, xv(idx_ds), yv(idx_ds), zv(idx_ds), ...
                          ls, 'Color', mc, 'LineWidth', lw, ...
                          'DisplayName', method_labels{m});
                end

                format_sphere_axes(ax);
            end
        else
            % === All boresights on one figure ===
            fig = figure('Name', sprintf('[%s] Boresight', exp_name), ...
                         'Position', [100, 100, 500, 500]);
            ax = axes(fig); hold(ax, 'on');

            draw_sphere(ax);
            draw_koz_cones(ax, koz_dirs, koz_angles, koz_names, koz_colors);

            % Goal and start markers
            d_ref = all_data{1};
            for b = 1:N_bs
                bs_vec_B = ref.boresight_vecs_B(b, :);
                bs_goal = quat_rotate(q_goal, bs_vec_B);

                bs_traj_ref = d_ref.(bs_names{b});
                x0 = squeeze(bs_traj_ref(ep, :, 1));
                y0 = squeeze(bs_traj_ref(ep, :, 2));
                z0 = squeeze(bs_traj_ref(ep, :, 3));
                v0 = find(~isnan(x0), 1);

                if b == 1
                    plot3(ax, bs_goal(1), bs_goal(2), bs_goal(3), ...
                          'p', 'Color', [0.85 0.65 0.0], 'MarkerSize', 18, ...
                          'MarkerFaceColor', [1 0.84 0], 'LineWidth', 1.2, ...
                          'HandleVisibility', 'off');
                    plot3(ax, x0(v0), y0(v0), z0(v0), ...
                          'o', 'Color', 'k', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
                          'LineWidth', 1.0, 'HandleVisibility', 'off');
                else
                    plot3(ax, bs_goal(1), bs_goal(2), bs_goal(3), ...
                          'p', 'Color', [0.85 0.65 0.0], 'MarkerSize', 18, ...
                          'MarkerFaceColor', [1 0.84 0], 'LineWidth', 1.2, ...
                          'HandleVisibility', 'off');
                    plot3(ax, x0(v0), y0(v0), z0(v0), ...
                          'o', 'Color', 'k', 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
                          'LineWidth', 1.5, 'HandleVisibility', 'off');
                end
            end

            % Method trajectories
            for m = 1:N_methods
                d_m = all_data{m};
                ci = color_indices(m);
                mc = method_colors(ci, :);
                ls = line_styles{ci};
                lw = line_widths(ci);

                for b = 1:N_bs
                    bs_traj = d_m.(bs_names{b});
                    x = squeeze(bs_traj(ep, :, 1));
                    y = squeeze(bs_traj(ep, :, 2));
                    z = squeeze(bs_traj(ep, :, 3));
                    valid = ~isnan(x);
                    xv = x(valid); yv = y(valid); zv = z(valid);
                    idx_ds = [1:ds_step:numel(xv), numel(xv)];

                    if b == 1
                        plot3(ax, xv(idx_ds), yv(idx_ds), zv(idx_ds), ...
                              ls, 'Color', mc, 'LineWidth', lw, ...
                              'DisplayName', method_labels{m});
                    else
                        plot3(ax, xv(idx_ds), yv(idx_ds), zv(idx_ds), ...
                              ls, 'Color', mc, 'LineWidth', lw, ...
                              'HandleVisibility', 'off');
                    end
                end
            end

            format_sphere_axes(ax);
        end
    end
end


% =========================================================================
%  Helper: draw unit sphere
% =========================================================================
function draw_sphere(ax)
    [Xs, Ys, Zs] = sphere(30);
    mesh(ax, Xs, Ys, Zs, ...
         'FaceAlpha', 0, 'EdgeColor', [0.75 0.75 0.75], ...
         'EdgeAlpha', 0.25, 'LineWidth', 0.3, ...
         'HandleVisibility', 'off');
end


% =========================================================================
%  Helper: draw KOZ cones
% =========================================================================
function draw_koz_cones(ax, koz_dirs, koz_angles, koz_names, koz_colors)
    for k = 1:numel(koz_names)
        dir_k = koz_dirs(k, :);
        dir_k = dir_k / norm(dir_k);
        c = koz_colors(mod(k-1, size(koz_colors,1)) + 1, :);
        plot_cone(ax, dir_k, koz_angles(k), c);
    end
end


% =========================================================================
%  Helper: format sphere axes
% =========================================================================
function format_sphere_axes(ax)
    axis(ax, 'equal');
    xlim(ax, [-1.3, 1.3]);
    ylim(ax, [-1.3, 1.3]);
    zlim(ax, [-1.3, 1.3]);
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'Z');
    set(ax, 'TickDir', 'out', 'Box', 'on');
    view(ax, 30, 25);
    legend(ax, 'Location', 'northeastoutside');
    hold(ax, 'off');
end


% =========================================================================
%  Helper: rotate vector by quaternion [w, x, y, z]
% =========================================================================
function v_out = quat_rotate(q, v)
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1-2*(y^2+z^2),  2*(x*y-w*z),    2*(x*z+w*y);
         2*(x*y+w*z),    1-2*(x^2+z^2),  2*(y*z-w*x);
         2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x^2+y^2)];
    v_out = (R * v(:))';
end


% =========================================================================
%  Helper: plot a KOZ cone (solid cone from origin to unit sphere)
% =========================================================================
function plot_cone(ax, direction, angle_deg, color)
    direction = direction / norm(direction);
    slant_height = 1.0;
    height = slant_height * cosd(angle_deg);
    radius = slant_height * sind(angle_deg);

    [theta, h] = meshgrid(linspace(0, 2*pi, 40), linspace(0, height, 20));
    x = radius * (h / height) .* cos(theta);
    y = radius * (h / height) .* sin(theta);
    z = h;

    R = rotation_matrix_from_vectors([0 0 1], direction);
    coords = R * [x(:)'; y(:)'; z(:)'];
    x_rot = reshape(coords(1, :), size(x));
    y_rot = reshape(coords(2, :), size(y));
    z_rot = reshape(coords(3, :), size(z));

    surf(ax, x_rot, y_rot, z_rot, ...
         'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.2, ...
         'HandleVisibility', 'off');
end


% =========================================================================
%  Helper: rotation matrix aligning vec1 to vec2
% =========================================================================
function R = rotation_matrix_from_vectors(vec1, vec2)
    a = vec1 / norm(vec1);
    b = vec2 / norm(vec2);
    v = cross(a, b);
    c = dot(a, b);
    if c <= -1 + 1e-10
        R = -eye(3);
    elseif c >= 1 - 1e-10
        R = eye(3);
    else
        s = norm(v);
        kmat = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        R = eye(3) + kmat + kmat^2 * ((1 - c) / (s^2));
    end
end
