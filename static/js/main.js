document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const fileUpload = document.getElementById('file-upload');
    const loadingContainer = document.getElementById('loading-container');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('error-container');
    const resultImage = document.getElementById('result-image');
    const downloadLink = document.getElementById('download-link');
    const decodeAnother = document.getElementById('decode-another');
    const tryAgain = document.getElementById('try-again');
    const errorMessage = document.getElementById('error-message');
    const resultCard = document.getElementById('result-container');
    const sstv_mode = document.getElementById('sstv-mode');
    const processingTime = document.getElementById('processing-time');
    const generateTestBtn = document.getElementById('generate-test');

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate file
        const file = fileUpload.files[0];
        if (!file) {
            showError('Please select a WAV file to decode.');
            return;
        }
        
        // Validate file type
        if (file.type !== 'audio/wav' && !file.name.toLowerCase().endsWith('.wav')) {
            showError('Only WAV files are supported. Please select a valid file.');
            return;
        }
        
        // Show loading indicator
        hideAllContainers();
        loadingContainer.classList.remove('d-none');
        
        // Create FormData and append file
        const formData = new FormData();
        formData.append('file', file);
        
        // Send file to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideAllContainers();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Show result
            resultCard.classList.remove('d-none');
            resultImage.src = data.image_path;
            sstv_mode.textContent = data.sstv_mode || 'Unknown';
            processingTime.textContent = data.processing_time || 'N/A';
            downloadLink.href = data.download_path;
        })
        .catch(error => {
            hideAllContainers();
            showError('An error occurred while processing your file. Please try again.');
            console.error('Error:', error);
        });
    });
    
    // Reset form for another file
    decodeAnother.addEventListener('click', function() {
        resetForm();
    });
    
    // Try again after error
    tryAgain.addEventListener('click', function() {
        resetForm();
    });
    
    // File input change - validate file type
    fileUpload.addEventListener('change', function() {
        const file = fileUpload.files[0];
        
        if (file) {
            if (file.type !== 'audio/wav' && !file.name.toLowerCase().endsWith('.wav')) {
                showError('Only WAV files are supported. Please select a valid file.');
                fileUpload.value = '';
            } else {
                errorContainer.classList.add('d-none');
            }
        }
    });
    
    // Generate and decode test SSTV signal
    generateTestBtn.addEventListener('click', function() {
        // Show loading indicator
        hideAllContainers();
        loadingContainer.classList.remove('d-none');
        
        // Call the server to generate a test file
        fetch('/generate_test')
            .then(response => response.json())
            .then(data => {
                hideAllContainers();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Show result
                resultCard.classList.remove('d-none');
                resultImage.src = data.image_path;
                sstv_mode.textContent = (data.sstv_mode || 'Unknown') + 
                                       (data.decoding_method ? ` (${data.decoding_method})` : '');
                processingTime.textContent = data.processing_time || 'N/A';
                downloadLink.href = data.download_path;
                
                // Handle alternative methods if available
                const altMethodsContainer = document.getElementById('alt-methods-container');
                const altMethodsRow = document.getElementById('alt-methods');
                
                // Clear previous alternatives
                altMethodsRow.innerHTML = '';
                
                // Check if we have alternative methods
                if (data.alternative_methods && data.alternative_methods.length > 0) {
                    // Show the container
                    altMethodsContainer.classList.remove('d-none');
                    
                    // Add each alternative method
                    data.alternative_methods.forEach(method => {
                        const methodCol = document.createElement('div');
                        methodCol.className = 'col-md-4';
                        
                        methodCol.innerHTML = `
                            <div class="card h-100">
                                <div class="card-header">
                                    <h6 class="mb-0">Method ${method.number}</h6>
                                </div>
                                <div class="card-body p-2 text-center">
                                    <img src="${method.image_path}" class="img-fluid rounded mb-2" style="max-height: 150px;" alt="Alternative decode">
                                    <div>
                                        <a href="${method.download_path}" class="btn btn-sm btn-outline-info" download>
                                            <i class="fas fa-download me-1"></i> Download
                                        </a>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        altMethodsRow.appendChild(methodCol);
                    });
                } else {
                    // Hide the container if no alternatives
                    altMethodsContainer.classList.add('d-none');
                }
                
                // Add a new button to download the test WAV file if available
                if (data.test_wav_path) {
                    const downloadContainer = downloadLink.parentElement;
                    
                    // Check if the download WAV button already exists
                    const existingWavBtn = document.getElementById('download-wav');
                    if (!existingWavBtn) {
                        const downloadWavBtn = document.createElement('a');
                        downloadWavBtn.id = 'download-wav';
                        downloadWavBtn.className = 'btn btn-info';
                        downloadWavBtn.href = data.test_wav_path;
                        downloadWavBtn.download = true;
                        downloadWavBtn.innerHTML = '<i class="fas fa-download me-2"></i> Download Test WAV File';
                        
                        // Insert after the download image button
                        downloadContainer.insertBefore(downloadWavBtn, downloadLink.nextSibling);
                        // Add spacing
                        const spacer = document.createElement('div');
                        spacer.className = 'mb-2';
                        downloadContainer.insertBefore(spacer, downloadWavBtn);
                    } else {
                        existingWavBtn.href = data.test_wav_path;
                    }
                }
            })
            .catch(error => {
                hideAllContainers();
                showError('An error occurred while generating the test file. Please try again.');
                console.error('Error:', error);
            });
    });
    
    // Helper functions
    function hideAllContainers() {
        loadingContainer.classList.add('d-none');
        resultCard.classList.add('d-none');
        errorContainer.classList.add('d-none');
    }
    
    function showError(message) {
        hideAllContainers();
        errorContainer.classList.remove('d-none');
        errorMessage.textContent = message;
    }
    
    function resetForm() {
        hideAllContainers();
        uploadForm.reset();
    }
});
