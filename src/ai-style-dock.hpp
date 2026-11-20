#pragma once

#include <QDockWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QSlider>
#include <QPushButton>
#include <QProgressBar>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsOpacityEffect>
#include <QPixmap>
#include <QImage>
#include <QTabWidget>
#include <QComboBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QFile>
#include <QDir>
#include <mutex>
#include <atomic>
#include <vector>

#include <obs-module.h>

struct ModelDef {
    QString name;
    QString type; // Checkpoint, LoRA, ControlNet, VAE
    QString filename;
    QString url;
    QString description;
    QString localPath; // Relative to models/
};

class AIStyleDock : public QDockWidget {
    Q_OBJECT

public:
    explicit AIStyleDock(QWidget *parent = nullptr);
    ~AIStyleDock();
    
    // Called from C++ to update the preview
    void updatePreview(const uint8_t* data, int width, int height);
    void updateStats(float fps, int dropped, float vram_mb);
    void updateProgress(float progress);
    void setStatus(const QString &status);
    void refreshTheme();
    
    // Connect to a filter source
    void connectToSource(obs_source_t *source);

    // Register a source as active (called by filters on create)
    void registerActiveSource(obs_source_t *source);
    void unregisterActiveSource(obs_source_t *source);

signals:
    void promptChanged(const QString &prompt);
    void strengthChanged(double strength);
    void stepsChanged(int steps);
    void presetSelected(const QString &preset);
    void generateClicked();

private slots:
    void onPromptEdited();
    void onStrengthSliderChanged(int value);
    void onStepsSliderChanged(int value);
    void onPresetClicked();
    void onRefreshStats();
    void pulseProgress();
    
    // Source Management
    void onSourceSelected(int index);
    void scanForSources();
    
    // Model Manager
    void onInstallModelClicked();
    void onDownloadProgress(qint64 bytesReceived, qint64 bytesTotal);
    void onDownloadFinished();

private:
    void setupUI();
    void setupControlTab();
    void setupManagerTab();
    void applyThemeStyle();
    void createAnimations();
    void applyPresetToFilter(const QString &preset);
    
    // Helper to find plugin sources
    void populateSourceList();
    void populateModelList();
    QString getModelInstallPath(const QString &type);
    bool isModelInstalled(const ModelDef &model);

    // UI Elements
    QTabWidget *mainTabs;
    QWidget *controlWidget;
    QWidget *managerWidget;
    
    // Source Selector
    QComboBox *sourceSelector;
    QPushButton *refreshSourcesBtn;

    // Control Tab Elements
    QLabel *previewLabel;
    QLabel *statusLabel;
    QLabel *fpsLabel;
    QLabel *vramLabel;
    QLabel *droppedLabel;
    
    QLineEdit *promptEdit;
    QSlider *strengthSlider;
    QLabel *strengthValueLabel;
    QSlider *stepsSlider;
    QLabel *stepsValueLabel;
    
    QProgressBar *progressBar;
    
    QPushButton *animeBtn;
    QPushButton *cineBtn;
    QPushButton *cyberBtn;
    QPushButton *zimageBtn;
    
    QPushButton *generateBtn;
    
    // Manager Tab Elements
    QTableWidget *modelTable;
    QProgressBar *downloadBar;
    QLabel *downloadStatusLabel;
    
    // Network
    QNetworkAccessManager *netManager;
    QNetworkReply *currentReply = nullptr;
    QFile *currentFile = nullptr;
    
    // Data
    QVector<ModelDef> availableModels;
    
    // Animations
    QPropertyAnimation *pulseAnim;
    QTimer *statsTimer;
    QTimer *progressTimer;
    
    // Connected source
    obs_source_t *connectedSource = nullptr;
    std::mutex previewMutex;
    
    // Preview data
    QPixmap currentPreview;
    
    // List of potential sources (Name -> weak_ptr equivalent not avail in C source, using name map)
    struct SourceItem {
        QString name;
        obs_source_t* source;
    };
    std::vector<SourceItem> detectedSources;
};

// Global instance for C access
extern AIStyleDock *g_aiStyleDock;

// C-compatible functions for use from filter code
extern "C" {
    void ai_dock_update_preview(const uint8_t* data, int width, int height);
    void ai_dock_update_stats(float fps, int dropped, float vram_mb);
    void ai_dock_update_progress(float progress);
    void ai_dock_set_status(const char* status);
    void ai_dock_connect_source(obs_source_t *source);
    
    // New: Allow filters to register themselves to the dock list
    void ai_dock_register_source(obs_source_t *source);
    void ai_dock_unregister_source(obs_source_t *source);
}
