function ok = place_legend_below_row(lgd, ax_row, gap)
%PLACE_LEGEND_BELOW_ROW  Centers a horizontal legend under a row of IRF
%panels and sets its vertical position just below that row's own x-tick
%labels, rather than a fixed offset guessed from the axes' Position.
%
%lgd    : the legend object (already created via legend(...)) to
%         reposition.
%ax_row : row vector of axes handles for one row of panels (e.g. the BLP
%         row of Figure 1) -- the legend is centered under the whole
%         row, not just its last panel, and cleared of whichever panel
%         needs the most tick-label room.
%gap    : additional fixed buffer, in normalized figure units, left
%         between the row's rendered tick labels and the top of the
%         legend, so it reads as clearly separated rather than just
%         barely clearing the axis.
%
%Each axes' TightInset(2) is the bottom margin (in the same normalized
%units as Position) that MATLAB reserves for that axes' own tick labels
%-- read here after a drawnow so it reflects what is actually rendered
%(current FontSize, tick value strings, etc.) instead of a constant
%tuned for one specific font size/figure size that silently stops being
%enough once, say, the panels' FontSize changes. The row's minimum (most
%conservative) panel is used so the legend clears every panel in the
%row, not just one.
%
%Callers should also install this as the containing figure's
%SizeChangedFcn (see the calling driver script for an example),
%since TightInset is a fraction of the axes' fixed, normalized Position,
%and that fraction is not resize-invariant: tick-label text is a fixed
%size in points, so it occupies a different fraction of a bigger vs.
%smaller figure (e.g. after maximizing a saved .fig to full screen). A
%legend position computed once at creation time can therefore drift back
%into overlapping the x-axis on resize; recomputing on every
%SizeChangedFcn keeps it clear regardless of window size. (This does
%mean a resize will undo a legend you dragged by hand afterwards --
%these figures are meant to be reproducible/paper-ready rather than
%hand-tweaked, so that tradeoff is intentional.)
%
%Returns ok=true unconditionally, purely so this can be chained with '&&'
%inside a single-expression anonymous SizeChangedFcn that repositions
%more than one legend (anonymous functions can't contain multiple
%statements) -- see callers.
drawnow;
row_bottom = min(arrayfun(@(a) a.Position(2) - a.TightInset(2), ax_row));
lgd.Position(1) = 0.5 - lgd.Position(3)/2;
lgd.Position(2) = row_bottom - gap;
ok = true;
end
