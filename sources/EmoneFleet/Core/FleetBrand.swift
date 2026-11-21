/// The manager's name, in one place.
///
/// Two strings because they are used for different jobs: the **command** is what
/// a person types, so it is short and unpunctuated; the **name** is what the tool
/// calls itself in help, in `--version` and in the dashboard title. Keeping them
/// here means the CLI, the window and the tests cannot drift apart.
public enum FleetBrand {
    /// The installed executable, as typed.
    public static let command = "ttlanmanager"

    /// The product's own name.
    public static let name = "Emone DSH LAN Manager"
}
