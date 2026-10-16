function plot_compare_violation(mc_dirs, varargin)
%PLOT_COMPARE_VIOLATION  Per-episode worst-case KOZ gap box plot.
%
%   plot_compare_violation(MC_DIRS)
%   plot_compare_violation(MC_DIRS, 'Experiments', {'exp1b','exp2a','exp2b'})
%
%   MC_DIRS is a struct array with fields:
%       .path   - run directory under outputs/, e.g. 'mc_party/2026-03-27/07-39-50'
%       .name   - display name, e.g. 'CAK-SMPC'
%       .color  - [r g b]
%
%   Produces one subplot per experiment.  Within each subplot, all methods
%   are shown side-by-side as grouped box plots so distributions can be
%   compared directly.

    p = inputParser;
    addRequired(p, 'mc_dirs');
    addParameter(p, 'Experiments', {}, @iscell);
    parse(p, mc_dirs, varargin{:});
    exp_filter = p.Results.Experiments;

    N_methods = numel(mc_dirs);

    % --- Discover experiments from first method ---
    base_dir = fullfile('outputs', mc_dirs(1).path, 'data');
    mat_files = dir(fullfile(base_dir, 'exp*.mat'));
    exp_names = sort(cellfun(@(f) f(1:end-4), {mat_files.name}, ...
                             'UniformOutput', false));
    if ~isempty(exp_filter)
        exp_names = exp_names(ismember(exp_names, exp_filter));
    end
    N_exp = numel(exp_names);

    % --- Compute per-episode min gap for every (method, experiment) ---
    min_gaps = cell(N_methods, N_exp);

    for m = 1:N_methods
        data_dir = fullfile('outputs', mc_dirs(m).path, 'data');
        for e = 1:N_exp
            mat_path = fullfile(data_dir, [exp_names{e}, '.mat']);
            if ~isfile(mat_path)
                fprintf('WARNING: %s not found for %s\n', ...
                        exp_names{e}, mc_dirs(m).name);
                min_gaps{m,e} = [];
                continue;
            end

            data = load(mat_path);
            N_ep = double(data.num_episodes);
            bs_names = cellstr(data.boresight_names);
            koz_dirs = data.koz_directions;
            koz_angles = data.koz_angles_deg(:);

            ep_min_gap = inf(N_ep, 1);
            for bi = 1:numel(bs_names)
                bs_traj = data.(bs_names{bi});          % (N_ep, T, 3)
                for ki = 1:size(koz_dirs, 1)
                    kd = koz_dirs(ki,:) / norm(koz_dirs(ki,:));
                    cos_sep = squeeze(sum( ...
                        bs_traj .* reshape(kd, 1, 1, 3), 3));
                    cos_sep = min(max(cos_sep, -1), 1);
                    sep_deg = acosd(cos_sep);
                    gap_deg = sep_deg - koz_angles(ki);
                    ep_min_gap = min(ep_min_gap, min(gap_deg, [], 2));
                end
            end
            min_gaps{m,e} = ep_min_gap;
        end
    end

    % --- Build grouped data for boxplot ---
    fig = figure('Position', [100 100 200*N_exp 320]);

    for e = 1:N_exp
        ax = subplot(1, N_exp, e); hold(ax, 'on');

        % Stack all methods into one vector + grouping variable
        all_vals = [];
        all_grp  = [];
        for m = 1:N_methods
            g = min_gaps{m,e};
            if isempty(g); continue; end
            all_vals = [all_vals; g(:)];          %#ok<AGROW>
            all_grp  = [all_grp; m * ones(numel(g), 1)]; %#ok<AGROW>
        end

        if isempty(all_vals); continue; end

        % boxplot with method colors
        bp = boxplot(ax, all_vals, all_grp, ...
                     'Labels', {mc_dirs.name}, ...
                     'Widths', 0.55, ...
                     'Symbol', '', ...       % hide outlier markers
                     'Whisker', Inf);        % whiskers to min/max

        % Color each box
        h_boxes = findobj(ax, 'Tag', 'Box');
        % boxplot draws boxes in reverse order
        for m = 1:min(numel(h_boxes), N_methods)
            idx = N_methods - m + 1;
            set(h_boxes(m), 'Color', mc_dirs(idx).color, 'LineWidth', 1.4);
        end
        h_med = findobj(ax, 'Tag', 'Median');
        for m = 1:min(numel(h_med), N_methods)
            idx = N_methods - m + 1;
            set(h_med(m), 'Color', mc_dirs(idx).color, 'LineWidth', 2.0);
        end
        h_whisk = findobj(ax, 'Tag', 'Whisker');
        for j = 1:numel(h_whisk)
            idx = N_methods - ceil(j/2) + 1;
            idx = max(1, min(idx, N_methods));
            set(h_whisk(j), 'Color', mc_dirs(idx).color, 'LineWidth', 1.0);
        end
        h_cap = findobj(ax, 'Tag', 'Adjacent');
        for j = 1:numel(h_cap)
            idx = N_methods - ceil(j/2) + 1;
            idx = max(1, min(idx, N_methods));
            set(h_cap(j), 'Color', mc_dirs(idx).color, 'LineWidth', 1.0);
        end

        % KOZ boundary
        xlims = xlim(ax);
        plot(ax, xlims, [0 0], ':', 'Color', [0.8 0 0], 'LineWidth', 1.5);

        label = strrep(exp_names{e}, 'exp', '');
        title(ax, sprintf('Scenario %s', label));
        if e == 1
            ylabel(ax, 'Min gap [deg]');
        end
        set(ax, 'XTickLabelRotation', 30, 'TickDir', 'out', 'Box', 'on');
        grid(ax, 'on'); set(ax, 'GridAlpha', 0.15);
        hold(ax, 'off');
    end
end
