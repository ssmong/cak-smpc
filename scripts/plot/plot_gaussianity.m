function plot_gaussianity(timestamp)
%PLOT_GAUSSIANITY  Q-Q plots of latent prediction residuals (Appendix figure).
%
%   plot_gaussianity('2026-03-07/12-34-06')
%
%   Reads gaussianity_data.mat and generates a 2x2 Q-Q plot grid.
%
%   Inputs:
%       timestamp - e.g. '2026-03-07/12-34-06'

    arguments
        timestamp (1,1) string
    end

    mat_path = fullfile('outputs/learning_party', ...
                        timestamp, 'gaussianity/gaussianity_data.mat');

    if ~isfile(mat_path)
        error('File not found: %s', mat_path);
    end

    data = load(mat_path);

    N           = double(data.num_samples);
    D           = double(data.latent_dim);
    K           = double(data.num_selected);
    qq_range    = double(data.qq_range);
    dims        = double(data.selected_dims);       % (1, K) 1-indexed
    skew_sel    = double(data.selected_skewness);   % (1, K)
    kurt_sel    = double(data.selected_kurtosis);   % (1, K)
    skew_mean   = double(data.skewness_mean);
    kurt_mean   = double(data.kurtosis_mean);
    qq_theo     = data.qq_theoretical;              % cell (1, K)
    qq_emp      = data.qq_empirical;                % cell (1, K)

    % --- Layout: 2x2 grid, plot first 4 dims ---
    n_plot = min(K, 4);
    n_cols = 2;
    n_rows = 2;

    % --- Colors ---
    c_data = [0.27 0.51 0.71];     % steelblue
    c_ref  = [0.80 0.20 0.20];     % red

    fig = figure('Position', [100 100 450*n_cols 415*n_rows]);

    for idx = 1:n_plot
        ax = subplot(n_rows, n_cols, idx);
        hold(ax, 'on');

        theo = double(qq_theo{idx});
        emp  = double(qq_emp{idx});

        % Data points
        scatter(ax, theo, emp, 4, c_data, 'filled', ...
                'MarkerFaceAlpha', 0.35);

        % Reference line
        plot(ax, [-qq_range, qq_range], [-qq_range, qq_range], ...
             '-', 'Color', c_ref, 'LineWidth', 1.5);

        % Formatting
        ticks = -floor(qq_range):1:floor(qq_range);
        xlim(ax, [-qq_range, qq_range]);
        ylim(ax, [-qq_range, qq_range]);
        set(ax, 'XTick', ticks, 'YTick', ticks);
        axis(ax, 'square');
        grid(ax, 'on');
        ax.GridAlpha = 0.2;

        % Bottom row only: add x-axis label
        if idx > n_cols
            xlabel(ax, 'Theoretical quantiles');
        end
        % Left column only: add y-axis label
        if mod(idx, n_cols) == 1
            ylabel(ax, 'Empirical quantiles');
        end
        title(ax, sprintf('Dim %d', dims(idx)));

        set(ax, 'TickDir', 'out');

        % Legend only on first subplot
        if idx == 1
            legend(ax, {'Residuals', 'N(0,1)'}, ...
                   'Location', 'northwest');
        end

        hold(ax, 'off');
    end

    % Hide unused subplots
    for idx = n_plot+1 : n_rows*n_cols
        ax = subplot(n_rows, n_cols, idx);
        set(ax, 'Visible', 'off');
    end

    % --- Console summary ---
    fprintf('\n=== Gaussianity Summary ===\n');
    fprintf('  Samples:       N = %d\n', N);
    fprintf('  Latent dim:    D = %d\n', D);
    fprintf('  Mean skewness: %.4f  (ideal: 0)\n', skew_mean);
    fprintf('  Mean kurtosis: %.4f  (ideal: 3)\n', kurt_mean);
    fprintf('\n  Selected dimensions:\n');
    fprintf('  %5s  %10s  %10s\n', 'Dim', 'Skewness', 'Kurtosis');
    fprintf('  %s\n', repmat('-', 1, 30));
    for i = 1:K
        fprintf('  %5d  %10.4f  %10.4f\n', dims(i), skew_sel(i), kurt_sel(i));
    end
    fprintf('===========================\n');
end