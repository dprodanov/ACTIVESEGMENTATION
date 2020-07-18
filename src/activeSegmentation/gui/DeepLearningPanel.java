package activeSegmentation.gui;

import activeSegmentation.ASCommon;
import activeSegmentation.IDeepLearning;
import activeSegmentation.deepLearning.UNetImplementation;
import activeSegmentation.learning.DeepLearningManager;
import activeSegmentation.prj.ProjectInfo;
import activeSegmentation.prj.ProjectManager;
import activeSegmentation.util.GuiUtil;
import weka.core.OptionHandler;
import weka.gui.GenericObjectEditor;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;

public class DeepLearningPanel extends Component implements Runnable, ASCommon, ActionListener, PropertyChangeListener {
    private JList<String> modelList;
    private GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();
    private String originalOptions;
    String originalClassifierName;
    private ProjectManager projectManager;
    private ProjectInfo projectInfo;
    final JFrame frame = new JFrame("DEEP LEARNING");
    JList<String> featureSelList;
    final ActionEvent SAVE_BUTTON_PRESSED = new ActionEvent(this, 1, "Save");
    DeepLearningManager deepLearningManager;
    Button openButton;
    Button saveButton;
    JFileChooser fc;


    public DeepLearningPanel(ProjectManager projectManager, DeepLearningManager deepLearningManager)  {
        this.projectManager = projectManager;
        this.deepLearningManager=deepLearningManager;
        this.projectInfo = projectManager.getMetaInfo();
        this.modelList = GuiUtil.model();
    }

    public void doAction(ActionEvent event)  {
        if (event == this.SAVE_BUTTON_PRESSED)     {
            //System.out.println(this.featureSelList.getSelectedIndex());
            this.projectInfo.setFeatureSelection((String)this.featureSelList.getSelectedValue());

            // System.out.println("in set classifiler");
            IDeepLearning testClassifier=setClassifier();

            if(testClassifier!=null) {
                IDeepLearning deepModel = new UNetImplementation();
                this.deepLearningManager.setClassifier(deepModel);
                this.projectManager.updateMetaInfo(this.projectInfo);
            }

        }
    }

    public void actionPerformed(ActionEvent e) {

        //Handle open button action.
        if (e.getSource() == openButton) {
            int returnVal = fc.showOpenDialog(DeepLearningPanel.this);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                //This is where a real application would open the file.
                System.out.println(file.getName());
            } else {
                System.out.println("oops");
            }
            //Handle save button action.
        } else if (e.getSource() == saveButton) {
            int returnVal = fc.showSaveDialog(DeepLearningPanel.this);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                //This is where a real application would save the file.
                System.out.println(file.getName());
            } else {
                System.out.println("oops 2");
            }
        }
    }

    public void run()  {
        this.frame.setDefaultCloseOperation(1);
        this.frame.getContentPane().setBackground(Color.GRAY);
        this.frame.setLocationRelativeTo(null);
        this.frame.setSize(800, 550);
        JPanel learningP = new JPanel();
        learningP.setLayout(null);
        learningP.setBackground(Color.GRAY);

        JPanel learningJPanel = new JPanel();
        learningJPanel.setBorder(BorderFactory.createTitledBorder("Select the model"));
        String[] models = {"UNet", "OtherModel"};
        JList list = new JList(models);
        JScrollPane scrollPane = new JScrollPane(list);
        learningJPanel.add(scrollPane);
        learningJPanel.setBounds(30, 30, 250, 60);


        JPanel options = new JPanel();
        options.setBorder(BorderFactory.createTitledBorder("Learning Options"));
        options.setBounds(30, 120, 250, 80);


        Checkbox transferLearning = new Checkbox("Transfer learning");
        options.add(transferLearning);
        JPanel resetJPanel = new JPanel();
        resetJPanel.setBackground(Color.GRAY);
        resetJPanel.setBounds(370, 120, 200, 80);
        resetJPanel.add(addButton("SAVE", null, 500, 120, 200, 50, this.SAVE_BUTTON_PRESSED));

        JPanel parametersPanel = new JPanel();
        parametersPanel.setBorder(BorderFactory.createTitledBorder("Learning Parameters"));
        parametersPanel.setBounds(370, 20, 200, 100);

        JLabel learningRateLabel = new JLabel("Learning rate:");
        JLabel numEpochsLabel = new JLabel("Number of epochs:");
        JLabel batchSizeLabel = new JLabel("Batch size:");


        JFormattedTextField learningRate = new JFormattedTextField();
        learningRate.setColumns(2);
        learningRate.addPropertyChangeListener("value", this);
        JFormattedTextField numEpochs = new JFormattedTextField();
        numEpochs.setColumns(2);
        numEpochs.addPropertyChangeListener("value", this);
        JFormattedTextField batchSize = new JFormattedTextField();
        batchSize.setColumns(2);
        batchSize.addPropertyChangeListener("value", this);




        learningRateLabel.setLabelFor(learningRateLabel);
        numEpochsLabel.setLabelFor(numEpochs);
        batchSizeLabel.setLabelFor(batchSize);

        parametersPanel.add(learningRateLabel);
        parametersPanel.add(learningRate);
        parametersPanel.add(numEpochsLabel);
        parametersPanel.add(numEpochs);
        parametersPanel.add(batchSizeLabel);
        parametersPanel.add(batchSize);

        JPanel importLabels = new JPanel();
        importLabels.setBorder(BorderFactory.createTitledBorder("Import labels"));
        JFileChooser fc = new JFileChooser();
        JButton openButton = new JButton("Import labels");
        JButton saveButton = new JButton("Save labels");

        fc.setVisible(true);
        fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        JPanel buttonPanel = new JPanel(); //use FlowLayout
        buttonPanel.add(openButton);
        buttonPanel.add(saveButton);

        openButton.setVisible(true);
        openButton.addActionListener(this);
        saveButton.addActionListener(this);
        File labels = new File("label");
        if(!labels.exists()){
            labels.mkdirs();
        }
        importLabels.add(buttonPanel);
//        importLabels.add(fc);
        importLabels.setBounds(400, 150, 300, 100);

        learningP.add(importLabels);
        learningP.add(parametersPanel);
        learningP.add(learningJPanel);
        learningP.add(resetJPanel);
        learningP.add(options);


        this.frame.add(learningP);
        this.frame.setVisible(true);
    }
//    public void propertyChange(PropertyChangeEvent e) {
//        Object source = e.getSource();
//        if (source == amountField) {
//            amount = ((Number)amountField.getValue()).doubleValue();
//        } else if (source == rateField) {
//            rate = ((Number)rateField.getValue()).doubleValue();
//        } else if (source == numPeriodsField) {
//            numPeriods = ((Number)numPeriodsField.getValue()).intValue();
//        }
//
//        double payment = computePayment(amount, rate, numPeriods);
//        paymentField.setValue(new Double(payment));
//    }

    private IDeepLearning setClassifier()
    {
        Object c = this.m_ClassifierEditor.getValue();
        String options = "";
        String[] optionsArray = ((OptionHandler)c).getOptions();
        System.out.println(originalOptions);
//        if ((c instanceof OptionHandler)) {
//            options = Utils.joinOptions(optionsArray);
//        }
//        if ((!this.originalClassifierName.equals(c.getClass().getName())) ||
//                (!this.originalOptions.equals(options))) {
//            try
//            {
//                AbstractClassifier cls = (AbstractClassifier)c.getClass().newInstance();
//                cls.setOptions(optionsArray);
//                return cls;
//            }
//            catch (Exception ex)
//            {
//                ex.printStackTrace();
//            }
//        }
        return null;
    }

    private JButton addButton(String label, ImageIcon icon, int x, int y, int width, int height, ActionEvent SAVE_BUTTON_PRESSED)
    {
        JButton button = new JButton(label, icon);
        button.setFont(labelFONT);
        button.setBorderPainted(false);
        button.setFocusPainted(false);
        button.setBackground(new Color(192, 192, 192));
        button.setForeground(Color.WHITE);
        button.setBounds(x, y, width, height);
        return button;
    }

    @Override
    public void propertyChange(PropertyChangeEvent propertyChangeEvent) {

    }
}