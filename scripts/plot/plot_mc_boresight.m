function plot_mc_boresight(run_dir, az, el, opts)
%PLOT_MC_BORESIGHT  Overlay boresight trajectories from MC runs.
%
%   plot_mc_boresight(RUN_DIR)
%   plot_mc_boresight(RUN_DIR, AZ, EL)
%   plot_mc_boresight(RUN_DIR, AZ, EL, N_ep=50)
%
%   RUN_DIR is the date/time subdirectory under outputs/,
%   e.g. 'mc_party/2026-03-09/05-03-21'.
%   AZ, EL are the azimuth and elevation for the view (default: 0, 30).
%   N_ep is the number of episodes to plot (default: all).

    arguments
        run_dir
        az  (1,1) double = 0
        el  (1,1) double = 30
        opts.N_ep (1,1) double = Inf
    end

    base_dir = fullfile('outputs', run_dir, 'data');
    if ~isfolder(base_dir)
        error('Directory not found: %s', base_dir);
    end

    % List .mat files
    files = dir(fullfile(base_dir, '*.mat'));
    if isempty(files)
        error('No .mat files found in %s', base_dir);
    end

    fprintf('\nAvailable data files:\n');
    for i = 1:numel(files)
        [~, name, ~] = fileparts(files(i).name);
        fprintf('  %d. %s\n', i, name);
    end
    fprintf('\n');
    idx = input('Select file number: ');
    if isempty(idx) || idx < 1 || idx > numel(files)
        error('Invalid selection.');
    end

    mat_filepath = fullfile(base_dir, files(idx).name);
    fprintf('Loading %s ...\n', mat_filepath);

    data = load(mat_filepath);

    bs_names   = cellstr(data.boresight_names);
    koz_names  = cellstr(data.koz_names);
    koz_dirs   = data.koz_directions;   % (M, 3)
    koz_angles = data.koz_angles_deg;   % (1, M) or (M,)
    q_goal     = data.q_goal(:)';       % (1, 4)
    N_ep_total = double(data.num_episodes);
    N_ep = min(opts.N_ep, N_ep_total);
    N_bs       = numel(bs_names);

    cmap_blue = winter;
    cmap_red  = autumn;
    bs_cmaps = {cmap_blue, cmap_red};

    % Representative colors for legend (midpoint of each colormap)
    legend_colors = {[0.15, 0.30, 0.72], ...   % blue
                     [0.82, 0.22, 0.12]};       % red-orange

    % Legend labels
    legend_labels = {'Boresight y_B', ...
                     'Boresight z_B'};

    % --- Figure ---
    fig = figure('Position', [100 100 500 500]);
    ax = axes(fig, 'Projection', 'perspective');
    hold(ax, 'on');

    % --- Wireframe sphere ---
    [Xs, Ys, Zs] = sphere(30);
    mesh(ax, Xs, Ys, Zs, ...
         'FaceColor', 'none', 'EdgeColor', [0.8 0.8 0.8], ...
         'EdgeAlpha', 0.3, 'LineWidth', 0.3);

    % --- KOZ cones ---
    koz_colors = [0.85 0.15 0.15;    % red
                  0.15 0.15 0.85;    % blue
                  0.85 0.55 0.10;    % orange
                  0.15 0.75 0.35];   % green
    for k = 1:numel(koz_names)
        dir_k = koz_dirs(k, :);
        dir_k = dir_k / norm(dir_k);
        angle_deg = koz_angles(k);
        c = koz_colors(mod(k-1, size(koz_colors,1)) + 1, :);
        plot_cone(ax, dir_k, angle_deg, c);
    end

    % --- Dummy handles for legend ---
    h_legend = gobjects(N_bs, 1);
    for b = 1:N_bs
        h_legend(b) = plot3(ax, nan, nan, nan, '-', ...
                            'Color', legend_colors{b}, 'LineWidth', 2.0);
    end

    % --- Episode trajectories ---
    for b = 1:N_bs
        bs_name = bs_names{b};
        bs_traj = data.(bs_name);           % (N_ep, T, 3)
        bs_vec_B = data.boresight_vecs_B(b, :);

        % Goal boresight in inertial frame
        bs_goal = quat_rotate(q_goal, bs_vec_B);

        cmap = bs_cmaps{min(b, numel(bs_cmaps))};
        N_cmap = size(cmap, 1);
        cidx = round(linspace(1, N_cmap, N_ep));

        for ep = 1:N_ep
            x = squeeze(bs_traj(ep, :, 1));
            y = squeeze(bs_traj(ep, :, 2));
            z = squeeze(bs_traj(ep, :, 3));
            valid = ~isnan(x);
            xv = x(valid); yv = y(valid); zv = z(valid);

            c_ep = cmap(cidx(ep), :);

            % Trajectory line
            plot3(ax, xv, yv, zv, '-', ...
                  'Color', [c_ep, 0.45], 'LineWidth', 1.5);

            % Start marker (filled circle per episode)
            if ~isempty(xv)
                plot3(ax, xv(1), yv(1), zv(1), 'o', ...
                      'Color', c_ep, 'MarkerSize', 2.5, ...
                      'MarkerFaceColor', c_ep);
            end
        end

        % Goal marker
        plot3(ax, bs_goal(1), bs_goal(2), bs_goal(3), ...
              'p', 'Color', [0.75 0.55 0.0], 'MarkerSize', 14, ...
              'MarkerFaceColor', [0.95 0.75 0.10], 'LineWidth', 1.0);
    end

    % --- Legend ---
    legend(h_legend, legend_labels, 'Location', 'northeast');

    % --- Axes formatting ---
    axis(ax, 'equal');
    xlim(ax, [-1.3, 1.3]);
    ylim(ax, [-1.3, 1.3]);
    zlim(ax, [-1.3, 1.3]);
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'Z');
    set(ax, 'TickDir', 'out', 'Box', 'on');
    view(ax, az, el);
    hold(ax, 'off');
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
         'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.2);
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