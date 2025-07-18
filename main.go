package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// #############################################################################
// ## STYLES
// #############################################################################

var (
	// Style for the focused text input
	focusedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))
	// Style for blurred (unfocused) text input
	blurredStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	// Style for the help text
	helpStyle = blurredStyle
	// Style for the main application container
	docStyle = lipgloss.NewStyle().Margin(1, 2)
	// Style for informational messages, defined as a render function for convenience.
	infoStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("28")).Render
)

// #############################################################################
// ## MESSAGES
// #############################################################################

// Represents a message that a new checkpoint has been found.
type newCheckpointMsg struct{ path string }

// Represents a message that an error occurred while finding a checkpoint.
type checkpointErrMsg struct{ err error }

// Represents a message that the episode loop has completed a cycle.
type episodeLoopFinishedMsg struct{}

// #############################################################################
// ## MODEL
// #############################################################################

// The main model for our TUI application. It holds the application's state.
type model struct {
	// The path to the most recent checkpoint.
	checkpointPath string
	// The path to the Python script to execute.
	scriptPath string
	// Context for managing the command's lifecycle.
	ctx context.Context
	// Function to cancel the context.
	cancel context.CancelFunc
	// Indicates whether the episode loop is currently running.
	looping bool
	// Spinner to show activity.
	spinner spinner.Model
	// Text input for the task configuration.
	textInput textinput.Model
	// Error that might have occurred.
	err error
}

// initialModel creates the starting state of our application.
func initialModel(scriptPath string) *model {
	// Configure the spinner
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	// Configure the text input for tasks
	ti := textinput.New()
	ti.Placeholder = `{"speed_flip": 10, "ball_hunt": 10, "shooting": 50}`
	ti.SetValue(`{"speed_flip": 10, "ball_hunt": 10, "shooting": 50}`)
	ti.Focus()
	ti.CharLimit = 256
	ti.Width = 80

	return &model{
		scriptPath: scriptPath,
		spinner:    s,
		textInput:  ti,
		looping:    false, // Start looping by default
	}
}

// #############################################################################
// ## TEA COMMANDS
// #############################################################################

// waitForNewCheckpoint is a command that watches for new checkpoints.
func waitForNewCheckpoint() tea.Msg {
	path, err := getMostRecentCheckpoint()
	if err != nil {
		return checkpointErrMsg{err}
	}
	return newCheckpointMsg{path}
}

// runEpisodeLoop is a command that executes the python script.
func runEpisodeLoop(ctx context.Context, scriptPath, checkpointPath, task string) tea.Cmd {
	return func() tea.Msg {
		// Command to execute the python script
		cmd := exec.CommandContext(ctx, "python", scriptPath, "--checkpoint", checkpointPath) // , "--task", task)

		// Run the command
		output, err := cmd.CombinedOutput()
		if ctx.Err() == context.Canceled {
			// The command was canceled by the user, not an error
			return episodeLoopFinishedMsg{}
		}
		if err != nil {
			log.Printf("Error running python script: %v\nOutput: %s", err, string(output))
			return checkpointErrMsg{fmt.Errorf("script error: %w", err)}
		}

		// Let the application know the loop finished
		return episodeLoopFinishedMsg{}
	}
}

// #############################################################################
// ## HELPER FUNCTIONS
// #############################################################################

// getMostRecentCheckpoint finds the latest checkpoint directory.
func getMostRecentCheckpoint() (string, error) {
	var mostRecentFile string
	var mostRecentTime time.Time

	// Check if the root directory exists
	root := "ray_results"
	if _, err := os.Stat(root); os.IsNotExist(err) {
		return "", fmt.Errorf("directory '%s' not found. Please run this from your project root", root)
	}

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err // Propagate errors
		}

		// We are only interested in directories with the "checkpoint_" prefix.
		if d.IsDir() && strings.HasPrefix(d.Name(), "checkpoint_") {
			info, err := d.Info()
			if err != nil {
				// Can't get info, so skip.
				return nil
			}
			if mostRecentFile == "" || info.ModTime().After(mostRecentTime) {
				mostRecentTime = info.ModTime()
				mostRecentFile = path
			}
		}
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("error walking directory: %w", err)
	}

	if mostRecentFile == "" {
		return "", fmt.Errorf("no checkpoint directories found in 'ray_results'")
	}

	absPath, err := filepath.Abs(mostRecentFile)
	if err != nil {
		return "", fmt.Errorf("could not get absolute path for checkpoint: %w", err)
	}

	return absPath, nil
}

// isValidJSON checks if a string is valid JSON.
func isValidJSON(s string) bool {
	var js map[string]any
	return json.Unmarshal([]byte(s), &js) == nil
}

// #############################################################################
// ## BUBBLETEA LIFECYCLE
// #############################################################################

// Init is the first function that will be called. It returns a command.
func (m *model) Init() tea.Cmd {
	// Start by finding the first checkpoint and ticking the spinner.
	return tea.Batch(waitForNewCheckpoint, m.spinner.Tick)
}

// Update handles all incoming messages and updates the model accordingly.
func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	switch msg := msg.(type) {
	// Handle key presses
	case tea.KeyMsg:
		// Don't process general keys if the input is focused
		if m.textInput.Focused() {
			goto handle_input
		}

		switch msg.String() {
		case "ctrl+c", "q":
			// Quit the application
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		case " ":
			// Toggle the episode loop on/off
			m.looping = !m.looping
			if m.looping && m.checkpointPath != "" {
				// If we start looping again, run the episode immediately
				m.ctx, m.cancel = context.WithCancel(context.Background())
				return m, runEpisodeLoop(m.ctx, m.scriptPath, m.checkpointPath, m.textInput.Value())
			} else if !m.looping && m.cancel != nil {
				// If we stop looping, cancel the running command
				m.cancel()
			}
		case "e":
			// Focus the text input to edit the task
			m.textInput.Focus()
			m.err = nil
		}

	// A new checkpoint was found
	case newCheckpointMsg:
		m.checkpointPath = msg.path
		if m.looping {
			// If we are in a loop, run the episode for the new checkpoint
			m.ctx, m.cancel = context.WithCancel(context.Background())
			return m, runEpisodeLoop(m.ctx, m.scriptPath, m.checkpointPath, m.textInput.Value())
		}

	// An error occurred finding a checkpoint
	case checkpointErrMsg:
		m.err = msg.err
		return m, nil

	// The episode loop has finished
	case episodeLoopFinishedMsg:
		// If we are still looping, look for the next checkpoint
		if m.looping {
			return m, waitForNewCheckpoint
		}

	// Spinner tick
	case spinner.TickMsg:
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

handle_input:
	// Handle text input updates separately
	var inputCmd tea.Cmd
	m.textInput, inputCmd = m.textInput.Update(msg)
	cmds = append(cmds, inputCmd)

	// Handle enter key specifically for the text input
	if key, ok := msg.(tea.KeyMsg); ok && key.String() == "enter" {
		if m.textInput.Focused() {
			if !isValidJSON(m.textInput.Value()) {
				m.err = fmt.Errorf("invalid JSON format for task")
			} else {
				m.err = nil
				m.textInput.Blur()
				// If looping, stop the current run and start a new one with the new task
				if m.looping {
					if m.cancel != nil {
						m.cancel()
					}
					// A new loop will be triggered by the episodeLoopFinishedMsg after the old one is cancelled.
				}
			}
		}
	}

	return m, tea.Batch(cmds...)
}

// View renders the UI.
func (m *model) View() string {
	var b strings.Builder

	// Title
	b.WriteString("🚀 Rocket League Agent Viewer 🚀\n\n")

	// Checkpoint Info
	if m.checkpointPath != "" {
		b.WriteString(fmt.Sprintf("Watching Checkpoint: %s\n", infoStyle(m.checkpointPath)))
	} else {
		b.WriteString(fmt.Sprintf("%s Searching for checkpoints...\n", m.spinner.View()))
	}

	// Status
	if m.looping {
		b.WriteString(fmt.Sprintf("%s Running episode... (Press space to pause)\n\n", m.spinner.View()))
	} else {
		b.WriteString("Paused. (Press space to resume)\n\n")
	}

	// Task Input
	b.WriteString("Agent Task (press 'e' to edit, 'enter' to submit):\n")
	b.WriteString(m.textInput.View() + "\n\n")

	// Help
	b.WriteString(helpStyle.Render("q: quit | space: pause/resume | e: edit task\n"))

	// Error message
	if m.err != nil {
		b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Render(fmt.Sprintf("\nError: %v", m.err)))
	}

	return docStyle.Render(b.String())
}

// #############################################################################
// ## MAIN
// #############################################################################

func main() {
	// Name of the python script to be executed.
	// This script should be in the same directory as the Go program.
	const pythonScriptName = "run_inference.py"

	// Check if the python script exists
	if _, err := os.Stat(pythonScriptName); os.IsNotExist(err) {
		log.Fatalf("Error: Python script '%s' not found. Please make sure it's in the same directory.", pythonScriptName)
	}

	// Make sure ray_results directory exists for the directory walk to work, even if empty
	if _, err := os.Stat("ray_results"); os.IsNotExist(err) {
		log.Println("Creating dummy 'ray_results/exp/checkpoint_000001' directory.")
		log.Println("Please replace with your actual checkpoint directories.")
		err := os.MkdirAll("ray_results/exp/checkpoint_0001", 0755)
		if err != nil {
			log.Fatalf("Failed to create dummy checkpoint directory: %v", err)
		}
	}

	p := tea.NewProgram(initialModel(pythonScriptName))
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}
}
