function plot_severity_boxplot()
%PLOT_SEVERITY_BOXPLOT  Violation severity box plot from MC JSON results.
%
%   Reads per-episode min_gap_deg from mc_validation_results.json for each
%   controller and plots a grouped box plot for Scenario 2b.

    % === Configuration: edit paths here ===
    methods = struct( ...
        'name',  {'NMPC',  'L-SMPC',  'UKF-SMPC',  'ASGP-SMPC',  'CAK-SMPC'}, ...
        'json',  { ...
            'mc_nmpc/2026-03-14/05-25-30', ...
            'mc_lsmpc/2026-03-17/03-22-14', ...
            'mc_ukf/2026-03-18/11-28-47', ...
            'mc_asgp/2026-03-24/01-43-36', ...
            'mc_party/2026-05-21/01-09-42'}, ...
        'exp_prefix', {'exp2', 'exp2', 'exp2', 'exp2', 'exp2'}, ...
        'color', { ...
            [0.8  0.0  0.0], ...   % NMPC red
            [0.8  0.0  0.6], ...   % L-SMPC magenta
            [0.0  0.6  0.0], ...   % UKF green
            [0.0  0.0  0.0], ...   % ASGP black
            [0.0  0.0  0.8]} ...   % CAK blue
    );

    N = numel(methods);
    all_gaps = cell(N, 1);

    for m = 1:N
        json_path = fullfile('outputs', methods(m).json, 'mc_validation_results.json');
        txt = fileread(json_path);
        data = jsondecode(txt);

        % Find the experiment key matching exp_prefix (handles suffixes like /alpha0.95/with_margin)
        fnames = fieldnames(data.results);
        % jsondecode converts '/' to nested structs or mangles names — flatten
        % Actually jsondecode with '/' in keys uses x_ prefix or underscores.
        % Safer: re-parse with regexp from raw text.
        exp_key = find_exp_key(txt, methods(m).exp_prefix);

        if isempty(exp_key)
            warning('No experiment matching "%s" found for %s', ...
                    methods(m).exp_prefix, methods(m).name);
            all_gaps{m} = [];
            continue;
        end

        % Navigate nested struct produced by jsondecode
        episodes = get_nested(data.results, exp_key, 'episodes');
        gaps = arrayfun(@(ep) ep.min_gap_deg, episodes);
        all_gaps{m} = gaps(:);
        fprintf('%s: %d episodes, median gap = %.2f deg\n', ...
                methods(m).name, numel(gaps), median(gaps));
    end

    % --- Plot (manual box plot — no Statistics Toolbox needed) ---
    fig = figure('Position', [100 100 480 340]);
    ax = axes(fig); hold(ax, 'on');

    box_w = 0.35;
    rng(42);
    for m = 1:N
        g = all_gaps{m};
        if isempty(g); continue; end
        sg = sort(g);
        n = numel(sg);
        q1 = sg(max(1, floor(0.25 * n)));
        q2 = sg(max(1, floor(0.50 * n)));
        q3 = sg(max(1, ceil(0.75 * n)));
        wlo = sg(1);
        whi = sg(end);
        c = methods(m).color;
        x = m;

        % Jittered scatter
        jitter = (rand(n, 1) - 0.5) * box_w * 0.7;
        scatter(ax, x + jitter, g, 8, c, 'filled', ...
                'MarkerFaceAlpha', 0.35, 'MarkerEdgeColor', 'none');

        % Box
        rectangle(ax, 'Position', [x - box_w/2, q1, box_w, q3 - q1], ...
                  'EdgeColor', c, 'LineWidth', 1.4, 'FaceColor', 'none');
        % Median
        plot(ax, [x - box_w/2, x + box_w/2], [q2 q2], '-', ...
             'Color', c, 'LineWidth', 2.0);
        % Whiskers
        plot(ax, [x x], [wlo q1], '-', 'Color', c, 'LineWidth', 1.0);
        plot(ax, [x x], [q3 whi], '-', 'Color', c, 'LineWidth', 1.0);
        cap_w = box_w * 0.5;
        plot(ax, [x-cap_w/2, x+cap_w/2], [wlo wlo], '-', 'Color', c, 'LineWidth', 1.0);
        plot(ax, [x-cap_w/2, x+cap_w/2], [whi whi], '-', 'Color', c, 'LineWidth', 1.0);
    end

    xlims = [0.5, N + 0.5];
    plot(ax, xlims, [0 0], ':', 'Color', [0.8 0 0], 'LineWidth', 1.5);
    xlim(ax, xlims);
    yl = ylim(ax);
    ylim(ax, [yl(1), 3]);
    set(ax, 'XTick', 1:N, 'XTickLabel', {methods.name});

    ylabel(ax, 'KOZ gap [deg]');
    set(ax, 'XTickLabelRotation', 30, 'TickDir', 'out', 'Box', 'on');
    grid(ax, 'on'); set(ax, 'GridAlpha', 0.15);
    hold(ax, 'off');

    % Save
    out_dir = fullfile('outputs', 'figures');
    if ~isfolder(out_dir); mkdir(out_dir); end
    exportgraphics(fig, fullfile(out_dir, 'severity_boxplot_2b.pdf'), ...
                   'ContentType', 'vector');
    fprintf('Saved to %s\n', fullfile(out_dir, 'severity_boxplot_2b.pdf'));
end


function key = find_exp_key(json_txt, prefix)
%FIND_EXP_KEY  Find the full experiment key in JSON that starts with prefix.
%   Handles keys like "exp2/alpha0.95/with_margin" when prefix is "exp2".
%   Excludes "exp2a" or "exp2b" when prefix is "exp2".
    tokens = regexp(json_txt, '"(exp2(?:/[^"]+)?)"', 'tokens');
    key = '';
    for i = 1:numel(tokens)
        candidate = tokens{i}{1};
        % Must start with prefix and NOT be exp2a, exp2b etc.
        if startsWith(candidate, prefix) && ...
           (numel(candidate) == numel(prefix) || candidate(numel(prefix)+1) == '/')
            key = candidate;
            return;
        end
    end
end


function val = get_nested(s, key, field)
%GET_NESTED  Navigate jsondecode struct using slash-separated key.
%   jsondecode converts "exp2/alpha0.95/with_margin" into nested structs
%   or mangles the field name. This function tries both approaches.

    parts = strsplit(key, '/');
    current = s;
    for i = 1:numel(parts)
        p = matlab.lang.makeValidName(parts{i});
        if isfield(current, p)
            current = current.(p);
        else
            % Try underscore-joined remainder
            remainder = strjoin(parts(i:end), '_');
            p2 = matlab.lang.makeValidName(remainder);
            if isfield(current, p2)
                current = current.(p2);
                break;
            else
                error('Cannot navigate to field "%s" in struct (tried "%s", "%s")', ...
                      key, p, p2);
            end
        end
    end
    val = current.(field);
end
