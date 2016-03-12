package dist.test;


import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
import java.util.UUID;

/**
 * Created by joanchen on 3/11/16.
 */
public class JGraph  {

    XYSeriesCollection dataset;

    HashMap<String, XYSeries> seriesMap;

    String x_axis;

    public JGraph(String[] seriesName, String x_title) {
        seriesMap = new HashMap<>();
        dataset = new XYSeriesCollection();
        for (String name : seriesName) seriesMap.put(name, new XYSeries(name, true));
        x_axis = x_title;
    }

    public void addToSeries(String ser, double x, double y) {
        XYSeries s = seriesMap.get(ser);
        s.add(x ,y);
    }

    private void createDataSet() {
        Set<String> keys = seriesMap.keySet();
        for(String k : keys) dataset.addSeries(seriesMap.get(k));
    }

    public void createChart(double x_min, double x_max) {
        createDataSet();
        JFreeChart chart = ChartFactory.createXYLineChart(x_axis + " vs error",
                x_axis, "error", dataset);

        System.out.print(dataset.getSeries(0).getItems());
        System.out.print(dataset.getSeries(1).getItems());

        XYPlot plot = (XYPlot) chart.getPlot();
        ValueAxis xAxis = plot.getDomainAxis();
        xAxis.setRange(x_min, x_max);
        ValueAxis yAxis = plot.getRangeAxis();
        yAxis.setRange(0.40, 0.80);

        File dir = new File("src/opt/test/results/graphs/" + x_axis + "/");
        if (!dir.isDirectory()) {
            dir.mkdir();
        }
        File imageFile = new File(dir.getAbsolutePath() + "/" + UUID.randomUUID() + ".png");
        int width = 640;
        int height = 480;

        try {
            ChartUtilities.saveChartAsPNG(imageFile, chart, width, height);
        } catch (IOException ex) {
            System.err.println(ex);
        }
    }


}
