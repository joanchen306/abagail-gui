package shared.test;

import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;

import shared.DataSet;
import shared.DataSetDescription;
import shared.reader.ArffDataSetReader;
import shared.reader.DataSetReader;
import shared.filt.ContinuousToDiscreteFilter;
import shared.filt.LabelSplitFilter;
import shared.reader.DataSetLabelBinarySeperator;

/**
 * A data set reader
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ArffDataSetReaderTest {
    /**
     * The test main
     * @param args ignored parameters
     */
    public static void main(String[] args) throws Exception {
        DataSetReader dsr = new ArffDataSetReader(new File("").getAbsolutePath() + "/src/opt/test/data/wine-test.arff");
        BufferedWriter writer = new BufferedWriter(new FileWriter("src/opt/test/wine-test.txt"));
        // read in the raw data
        DataSet ds = dsr.read();
        for(int i = 0; i < ds.size(); i++){
        	System.out.println(i + ": " + ds.get(i));
        	writer.write(ds.get(i).toString());
        	writer.newLine();
        }
        writer.close();
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(ds);
        ContinuousToDiscreteFilter ctdf = new ContinuousToDiscreteFilter(10);
        ctdf.filter(ds);
        DataSetLabelBinarySeperator.seperateLabels(ds);
        System.out.println("-----------------------------------------------");
        //System.out.println(ds);
        //System.out.println(new DataSetDescription(ds));
        System.out.println("/src/opt/test/data/wine-train.arff");
    }
}
