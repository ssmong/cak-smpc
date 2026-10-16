function plot_compare_timeseries(run_id, opts)
%   Options:
%       KOZFigure  - logical, plot standalone KOZ figure (default: false)
%PLOT_COMPARE_TIMESERIES  Overlay time-series diagnostics from a comparison run.
%
%   plot_compare_timeseries(RUN_ID)
%
%   For each experiment, produces a figure with 3 subplots:
%     Top:    Attitude error [deg] vs time
%     Middle: Angular velocity norm [deg/s] vs time
%     Bottom: KOZ minimum gap [deg] vs time  (negative = violation)
%
%   Uses the same controller ordering, colors, and styles as
%   plot_compare_boresight.m.
%
%   Inputs:
%       RUN_ID  - date/time string, e.g. '2026-03-17/10-30-00'
%
%   Example:
%       plot_compare_timeseries('2026-03-17/10-30-00');

    arguments
        run_id
        opts.KOZFigure (1,1) logical = false
    end

    % --- Fixed controller order and display names ---
    controller_order = {'nmpc', 'lsmpc', 'ukf_smpc', 'asgp_fast', 'party'};
    display_names = containers.Map( ...
        {'nmpc', 'lsmpc', 'ukf_smpc', 'asgp_fast', 'party'}, ...
        {'NMPC', 'L-SMPC', 'UKF-SMPC', 'ASGP-SMPC', 'CAK-SMPC'});

    % --- Method colors & styles (same as plot_compare_boresight) ---
    % party (ours) = blue solid; baselines = dashed/dotted, distinct colors
    method_colors = [
        1  0  0;   % nmpc      — 'r' red
        1.0, 0.6, 0.7;       % lsmpc
        0  0.8  0;   % ukf_smpc
        0  0  0;   % asgp_smpc — 'k' black
        0  0  1;   % party     — 'b' blue (ours)
    ];
    line_styles = {'-', '--', '-.', ':', '-'};
    line_widths = [1.75, 1.75, 1.75, 1.75, 1.75];

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
        n_ctrl = 0;
        for c = 1:numel(controller_order)
            mat_path = fullfile(base_path, exp_names{i}, controller_order{c}, ...
                                'data', 'sim_data.mat');
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
        ctrl_keys = {};
        sim_data_all = {};
        method_labels = {};
        color_indices = [];

        for c = 1:numel(controller_order)
            ctrl_key = controller_order{c};
            mat_path = fullfile(exp_path, ctrl_key, 'data', 'sim_data.mat');
            if isfile(mat_path)
                ctrl_keys{end+1} = ctrl_key; %#ok<AGROW>
                sim_data_all{end+1} = load(mat_path); %#ok<AGROW>
                method_labels{end+1} = display_names(ctrl_key); %#ok<AGROW>
                color_indices(end+1) = c; %#ok<AGROW>
            end
        end

        N_methods = numel(ctrl_keys);
        if N_methods < 1
            fprintf('Skipping %s: no controllers found.\n', exp_name);
            continue;
        end

        fprintf('\nPlotting %s (%d controller(s): %s)\n', ...
                exp_name, N_methods, strjoin(method_labels, ', '));

        % Check that timeseries fields exist
        ref = sim_data_all{1};
        has_timeseries = isfield(ref, 'att_err_deg') && ...
                         isfield(ref, 'omega_norm') && ...
                         isfield(ref, 'koz_min_gap_deg');

        if ~has_timeseries
            fprintf('  WARNING: sim_data.mat lacks timeseries fields.\n');
            fprintf('  Re-run 3_run_comparison.py with updated code.\n');
            fprintf('  Computing from raw state data...\n');
        end

        % Load KOZ info for annotation
        koz_info_loaded = false;
        for c = 1:numel(controller_order)
            bs_path = fullfile(exp_path, controller_order{c}, 'data', 'boresight_data.mat');
            if isfile(bs_path)
                bs_data = load(bs_path);
                koz_info_loaded = true;
                break;
            end
        end

        % =============================================================
        %  Figure: 3 subplots (att_err, omega_norm, koz_gap)
        % =============================================================
        fig = figure('Name', sprintf('[%s] Timeseries', exp_name), ...
                     'Position', [100, 100, 400, 550]);

        y_labels = {'Error [deg]', ...
                    '||u|| [Nm]', ...
                    'KOZ gap [deg]'};

        % Layout: top two subplots each 1 unit, bottom 2 units (total 4)
        gap = 0.08; margin_b = 0.08; margin_t = 0.04; margin_lr = 0.12;
        total_h = 1 - margin_b - margin_t - 2*gap;
        h_small = total_h / 4;
        h_large = 2 * h_small;

        ax_arr = gobjects(3, 1);
        ax_arr(1) = axes('Position', [margin_lr, margin_b + h_large + 2*gap + h_small, 1-2*margin_lr, h_small]); hold on;
        ax_arr(2) = axes('Position', [margin_lr, margin_b + h_large + gap, 1-2*margin_lr, h_small]); hold on;
        ax_arr(3) = axes('Position', [margin_lr, margin_b, 1-2*margin_lr, h_large]); hold on;

        for m = 1:N_methods
            sd = sim_data_all{m};
            t = sd.time(:);
            ci = color_indices(m);
            mc = method_colors(mod(ci-1, size(method_colors,1)) + 1, :);
            ls = line_styles{mod(ci-1, numel(line_styles)) + 1};
            lw = line_widths(mod(ci-1, numel(line_widths)) + 1);

            if has_timeseries
                att_err = sd.att_err_deg(:);
                ctrl_norm = vecnorm(sd.control, 2, 2);
                koz_gap = sd.koz_min_gap_deg(:);
            else
                % Fallback: compute from state
                state = sd.state;  % (T, 7)

                % Attitude error from q_goal (load from boresight_data)
                if koz_info_loaded
                    q_goal = bs_data.q_goal(:)';
                else
                    q_goal = state(end, 1:4);  % fallback to final state
                end

                quats = state(:, 1:4);
                qn = sqrt(sum(quats.^2, 2));
                quats = quats ./ qn;
                qg = q_goal / norm(q_goal);
                dots = abs(quats * qg');
                dots = min(dots, 1.0);
                att_err = rad2deg(2 * acos(dots));

                if isfield(sd, 'control')
                    ctrl_norm = vecnorm(sd.control, 2, 2);
                else
                    ctrl_norm = zeros(size(t));
                end

                % KOZ gap: compute from boresight data if available
                if koz_info_loaded
                    koz_dirs = bs_data.koz_directions;
                    koz_ang = bs_data.koz_angles_deg;
                    bs_names = cellstr(bs_data.boresight_names);
                    bs_vecs_B = bs_data.boresight_vecs_B;

                    T_len = size(state, 1);
                    koz_gap = inf(T_len, 1);
                    for ki = 1:size(koz_dirs, 1)
                        kd = koz_dirs(ki, :) / norm(koz_dirs(ki, :));
                        for bi = 1:size(bs_vecs_B, 1)
                            bv = bs_vecs_B(bi, :);
                            for ti = 1:T_len
                                q = quats(ti, :);
                                bs_N = quat_rotate(q, bv);
                                sep = rad2deg(acos(min(max(dot(bs_N, kd), -1), 1)));
                                gap_val = sep - koz_ang(ki);
                                koz_gap(ti) = min(koz_gap(ti), gap_val);
                            end
                        end
                    end
                else
                    koz_gap = zeros(size(t));
                end
            end

            % --- Plot ---
            plot(ax_arr(1), t, att_err, ls, 'Color', mc, 'LineWidth', lw, ...
                 'DisplayName', method_labels{m});
            plot(ax_arr(2), t, ctrl_norm, ls, 'Color', mc, 'LineWidth', lw, ...
                 'DisplayName', method_labels{m});
            koz_angle = 20;
            plot(ax_arr(3), t, koz_gap + koz_angle, ls, 'Color', mc, 'LineWidth', lw, ...
                 'HandleVisibility', 'off');
        end
        
        ylim(ax_arr(1), [0, 180]);
        ylim(ax_arr(2), [0, 0.9]);

        % --- KOZ threshold line ---
        t_range = xlim(ax_arr(3));
        plot(ax_arr(3), t_range, [koz_angle, koz_angle], ':', 'Color', [0.8 0 0], ...
             'LineWidth', 2.0, 'DisplayName', 'KOZ limit');
        ylim(ax_arr(3), [0, 80]);

        % --- Formatting ---
        for sp = 1:3
            ax = ax_arr(sp);
            ylabel(ax, y_labels{sp});
            set(ax, 'TickDir', 'out', 'Box', 'on');
            xlim(ax, 'tight');
            grid(ax, 'on');
            set(ax, 'GridAlpha', 0.15);

            if sp < 3
                set(ax, 'XTickLabel', []);
            else
                xlabel(ax, 'Time [s]');
            end
        end

        % Legend
        legend(ax_arr(1), 'Location', 'northeast', 'NumColumns', 1);
        legend(ax_arr(3), 'Location', 'northeast');

        % Link x-axes
        linkaxes(ax_arr, 'x');

        % --- Standalone KOZ angle figure (optional) ---
        if opts.KOZFigure
        fig_koz = figure('Name', sprintf('[%s] KOZ angle', exp_name), ...
                         'Position', [600 100 100 100]);
        ax_koz = axes(fig_koz); hold(ax_koz, 'on');

        for m = 1:N_methods
            sd = sim_data_all{m};
            t = sd.time(:);
            ci = color_indices(m);
            mc = method_colors(mod(ci-1, size(method_colors,1)) + 1, :);
            ls = line_styles{mod(ci-1, numel(line_styles)) + 1};
            lw = line_widths(mod(ci-1, numel(line_widths)) + 1);

            if has_timeseries
                koz_gap = sd.koz_min_gap_deg(:);
            else
                koz_gap = zeros(size(t));
            end
            plot(ax_koz, t, koz_gap + koz_angle, ls, 'Color', mc, 'LineWidth', lw);
        end
        t_range = xlim(ax_koz);
        plot(ax_koz, t_range, [koz_angle, koz_angle], ':', 'Color', [0.8 0 0], ...
             'LineWidth', 2.0);
        ylim(ax_koz, [0, 50]);
        xlim(ax_koz, [0, 50]);
        set(ax_koz, 'TickDir', 'out', 'Box', 'on');
        grid(ax_koz, 'on');
        set(ax_koz, 'GridAlpha', 0.15);
        hold(ax_koz, 'off');
        end  % if opts.KOZFigure

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
